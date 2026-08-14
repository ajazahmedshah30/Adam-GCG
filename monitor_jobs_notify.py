#!/usr/bin/env python3
import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import parse, request


ATTACKS = ("STATIC", "SEMI_DYNAMIC", "DYNAMIC")
SUPPORTED_MODES = {"adam_a40", "momentum_a40", "eval_adam_a40", "eval_momentum_a40"}
FINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "NODE_FAIL",
    "BOOT_FAIL",
}


@dataclass
class JobSpec:
    job_id: str
    mode: str
    label: str


def run_command(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(args, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def parse_job_spec(raw: str) -> JobSpec:
    parts = raw.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid --job format: {raw}. Use JOB_ID:MODE[:LABEL].")
    job_id, mode = parts[0], parts[1]
    label = parts[2] if len(parts) > 2 else f"{mode}:{job_id}"
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    return JobSpec(job_id=job_id, mode=mode, label=label)


def read_tail(path: Path, max_bytes: int = 65536) -> str:
    if not path.exists() or not path.is_file():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        return fh.read().decode("utf-8", errors="replace")


def get_scheduler_status(job_id: str) -> dict[str, str]:
    result = {"state": "UNKNOWN", "elapsed": "n/a", "node_or_reason": "n/a", "job_name": ""}

    rc, out, _ = run_command(["squeue", "-j", job_id, "-h", "-o", "%i|%j|%T|%M|%R"])
    if rc == 0 and out.strip():
        parts = out.strip().split("|", 4)
        if len(parts) == 5:
            _, name, state, elapsed, reason = parts
            result.update({
                "job_name": name,
                "state": state,
                "elapsed": elapsed,
                "node_or_reason": reason,
            })

    rc, out, _ = run_command([
        "sacct", "-j", job_id,
        "--format=JobIDRaw,JobName,State,Elapsed,NodeList",
        "-n", "-P",
    ])
    if rc == 0:
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) != 5:
                continue
            raw_id, name, state, elapsed, node = parts
            if raw_id != job_id:
                continue
            if result["job_name"] == "":
                result["job_name"] = name
            if state:
                result["state"] = state
            if elapsed:
                result["elapsed"] = elapsed
            if node and node != "None assigned":
                result["node_or_reason"] = node
            break
    return result


def get_attack_logs(repo_root: Path, spec: JobSpec) -> dict[str, Path]:
    if spec.mode == "adam_a40":
        log_dir = repo_root / "logs_a40"
        return {attack: log_dir / f"{attack}_{spec.job_id}.log" for attack in ATTACKS}
    return {attack: repo_root / f"{attack}.log" for attack in ATTACKS}


def get_active_attack(logs: dict[str, Path]) -> tuple[str | None, Path | None, str]:
    latest_attack = None
    latest_path = None
    latest_mtime = -1.0
    for attack, path in logs.items():
        if path.exists():
            mtime = path.stat().st_mtime
            if mtime >= latest_mtime:
                latest_mtime = mtime
                latest_attack = attack
                latest_path = path
    content = read_tail(latest_path) if latest_path else ""
    return latest_attack, latest_path, content


def get_active_evaluation_log(repo_root: Path, spec: JobSpec) -> tuple[str | None, Path | None, str]:
    log_dir = repo_root / "logs_eval_a40"
    if spec.mode == "eval_adam_a40":
        prefix = "eval_adam_"
    else:
        prefix = "eval_momentum_"

    candidates = sorted(
        log_dir.glob(f"{prefix}*_{spec.job_id}.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None, None, ""

    latest_path = candidates[0]
    stem = latest_path.stem
    suffix = stem[len(prefix):]
    suffix = re.sub(rf"_{re.escape(spec.job_id)}$", "", suffix)

    attack_label = "unknown"
    task_name = ""
    for attack_name in ("static", "semi-dynamic", "dynamic"):
        if suffix.startswith(f"{attack_name}_"):
            attack_label = attack_name.upper().replace("-", "_")
            task_name = suffix[len(attack_name) + 1:]
            break

    if task_name:
        attack_label = f"{attack_label}:{task_name}"
    return attack_label, latest_path, read_tail(latest_path)


def parse_progress(text: str) -> dict[str, str]:
    info: dict[str, str] = {}

    match = re.search(r"Resuming from step\s+(\d+)", text)
    if match:
        info["resume_step"] = match.group(1)

    epoch_matches = re.findall(r"Current Epoch:\s*(\d+)/(\d+)", text)
    if epoch_matches:
        step, total = epoch_matches[-1]
        info["step"] = step
        info["total"] = total
    else:
        data_matches = re.findall(r"Current Data:\s*(\d+)/(\d+)", text)
        if data_matches:
            step, total = data_matches[-1]
            info["step"] = step
            info["total"] = total
        prog_matches = re.findall(r"(\d+)/(\d+)", text) if "step" not in info else []
        if prog_matches:
            cur, total = prog_matches[-1]
            if "resume_step" in info:
                effective = int(info["resume_step"]) + int(cur)
                full_total = int(info["resume_step"]) + int(total)
                info["step"] = str(effective)
                info["total"] = str(full_total)
                info["step_remaining_view"] = f"{cur}/{total}"
            else:
                info["step"] = cur
                info["total"] = total

    loss_matches = re.findall(r"Loss:([0-9eE.+-]+)", text)
    if loss_matches:
        info["loss"] = loss_matches[-1]

    early = re.search(r"Early stopping at step\s+(\d+)", text)
    if early:
        info["note"] = f"early stopped at step {early.group(1)}"
    elif "Loading checkpoint..." in text:
        info["note"] = "resumed from checkpoint"

    return info


def get_stream_paths(repo_root: Path, spec: JobSpec) -> tuple[Path, Path]:
    if spec.mode == "adam_a40":
        return repo_root / f"adam_a40.{spec.job_id}.out", repo_root / f"adam_a40.{spec.job_id}.err"
    if spec.mode == "momentum_a40":
        return repo_root / f"mgcg_a40.{spec.job_id}.out", repo_root / f"mgcg_a40.{spec.job_id}.err"
    if spec.mode == "eval_adam_a40":
        return repo_root / f"eval_adam_a40.{spec.job_id}.out", repo_root / f"eval_adam_a40.{spec.job_id}.err"
    return repo_root / f"eval_momentum_a40.{spec.job_id}.out", repo_root / f"eval_momentum_a40.{spec.job_id}.err"


def summarize_job_data(repo_root: Path, spec: JobSpec) -> dict[str, str]:
    sched = get_scheduler_status(spec.job_id)
    if spec.mode in {"eval_adam_a40", "eval_momentum_a40"}:
        attack, attack_path, attack_text = get_active_evaluation_log(repo_root, spec)
    else:
        logs = get_attack_logs(repo_root, spec)
        attack, attack_path, attack_text = get_active_attack(logs)
    progress = parse_progress(attack_text)
    out_path, err_path = get_stream_paths(repo_root, spec)
    err_tail = read_tail(err_path, 8192).strip()
    out_tail = read_tail(out_path, 8192).strip()

    data = {
        "label": spec.label,
        "job_id": spec.job_id,
        "state": sched["state"],
        "elapsed": sched["elapsed"],
        "node_or_reason": sched["node_or_reason"],
        "attack": attack or "unknown",
        "step": f"{progress['step']}/{progress['total']}" if "step" in progress and "total" in progress else "-",
        "resume_step": progress.get("resume_step", "-"),
        "resumed_progress_view": progress.get("step_remaining_view", "-"),
        "loss": progress.get("loss", "-"),
        "note": progress.get("note", ""),
        "log_file": str(attack_path) if attack_path else "-",
        "stdout_tail": "-",
        "stderr_tail": "-",
    }
    if err_tail:
        data["stderr_tail"] = err_tail.splitlines()[-1]
    elif out_tail:
        data["stdout_tail"] = out_tail.splitlines()[-1]
    return data


def send_email(recipient: str, subject: str, body: str) -> tuple[bool, str]:
    mail_cmd = shutil.which("mail") or shutil.which("mailx")
    if not mail_cmd:
        return False, "mail command not found"
    proc = subprocess.run([mail_cmd, "-s", subject, recipient], input=body, text=True, capture_output=True)
    combined = "\n".join(part for part in (proc.stdout, proc.stderr) if part).strip()
    failure_markers = ("message not sent", "sendmail", "dead.letter", "No such file or directory")
    if proc.returncode == 0 and not any(marker in combined for marker in failure_markers):
        return True, "email sent via mail/mailx"
    return False, combined or "mail delivery failed"


def split_telegram_text(body: str, max_chars: int = 3200) -> list[str]:
    lines = body.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        addition = len(line) + (1 if current else 0)
        if current and current_len + addition > max_chars:
            chunks.append("\n".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += addition

    if current:
        chunks.append("\n".join(current))
    return chunks or [body]


def send_telegram(bot_token: str, chat_id: str, body: str) -> tuple[bool, str]:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    chunks = split_telegram_text(body)
    details: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        prefix = f"[{idx}/{len(chunks)}]\n" if len(chunks) > 1 else ""
        payload = {
            "chat_id": chat_id,
            "text": f"<pre>{html.escape(prefix + chunk)}</pre>",
            "parse_mode": "HTML",
        }
        data = parse.urlencode(payload).encode()
        try:
            with request.urlopen(request.Request(url, data=data), timeout=20) as resp:
                response_payload = resp.read().decode("utf-8", errors="replace")
            details.append(response_payload[:120])
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
    return True, "; ".join(details)


def send_slack(webhook_url: str, body: str) -> tuple[bool, str]:
    payload = json.dumps({"text": body}).encode("utf-8")
    req = request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=20) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
        return True, payload[:200]
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def deliver(args: argparse.Namespace, subject: str, body: str) -> tuple[bool, str]:
    attempts: list[str] = []
    if args.email:
        ok, detail = send_email(args.email, subject, body)
        attempts.append(f"email: {detail}")
        if ok:
            return True, "; ".join(attempts)

    bot_token = args.telegram_bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = args.telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if bot_token and chat_id:
        ok, detail = send_telegram(bot_token, chat_id, body)
        attempts.append(f"telegram: {detail}")
        if ok:
            return True, "; ".join(attempts)

    webhook_url = args.slack_webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if webhook_url:
        ok, detail = send_slack(webhook_url, body)
        attempts.append(f"slack: {detail}")
        if ok:
            return True, "; ".join(attempts)

    print(body)
    return False, "; ".join(attempts) if attempts else "no delivery target configured; printed to stdout"


def format_row(widths: dict[str, int], row: dict[str, str]) -> str:
    return (
        f"{row['label']:<{widths['label']}}  "
        f"{row['state']:<{widths['state']}}  "
        f"{row['attack']:<{widths['attack']}}  "
        f"{row['step']:<{widths['step']}}  "
        f"{row['elapsed']:<{widths['elapsed']}}"
    )


def format_separator(widths: dict[str, int], columns: tuple[str, ...]) -> str:
    return "  ".join("-" * widths[column] for column in columns)


def format_field_table(rows: list[tuple[str, str]]) -> list[str]:
    field_width = max(len("Field"), *(len(field) for field, _ in rows))
    value_width = max(len("Value"), *(len(value) for _, value in rows))
    header = f"{'Field':<{field_width}}  {'Value':<{value_width}}"
    separator = f"{'-' * field_width}  {'-' * value_width}"
    lines = [header, separator]
    for field, value in rows:
        lines.append(f"{field:<{field_width}}  {value:<{value_width}}")
    return lines


def build_message(repo_root: Path, specs: list[JobSpec]) -> tuple[str, str]:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    subject = f"Prajna Job Summary {timestamp}"
    rows = [summarize_job_data(repo_root, spec) for spec in specs]
    widths = {
        "label": max(len("Job"), *(len(row["label"]) for row in rows)),
        "state": max(len("State"), *(len(row["state"]) for row in rows)),
        "attack": max(len("Attack"), *(len(row["attack"]) for row in rows)),
        "step": max(len("Step"), *(len(row["step"]) for row in rows)),
        "elapsed": max(len("Elapsed"), *(len(row["elapsed"]) for row in rows)),
    }

    blocks = [
        f"Prajna job summary",
        f"Time: {timestamp}",
        "",
        format_row(widths, {
            "label": "Job",
            "state": "State",
            "attack": "Attack",
            "step": "Step",
            "elapsed": "Elapsed",
        }),
        format_separator(widths, ("label", "state", "attack", "step", "elapsed")),
    ]
    for row in rows:
        blocks.append(format_row(widths, row))
    for row in rows:
        blocks.append("")
        blocks.append(f"{row['label']} Details")
        blocks.extend(format_field_table([
            ("job_id", row["job_id"]),
            ("state", row["state"]),
            ("elapsed", row["elapsed"]),
            ("node/reason", row["node_or_reason"]),
            ("attack", row["attack"]),
            ("log_file", row["log_file"]),
            ("step", row["step"]),
            ("resumed_progress_view", row["resumed_progress_view"]),
            ("resumed_from_step", row["resume_step"]),
            ("latest_loss", row["loss"]),
            ("note", row["note"] or "-"),
            ("stdout_tail", row["stdout_tail"]),
            ("stderr_tail", row["stderr_tail"]),
        ]))
    return subject, "\n".join(blocks)


def all_finished(specs: list[JobSpec]) -> bool:
    for spec in specs:
        if get_scheduler_status(spec.job_id)["state"] not in FINAL_STATES:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor Slurm jobs and send status notifications.")
    parser.add_argument("--job", action="append", required=True, help="JOB_ID:MODE[:LABEL]")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between updates")
    parser.add_argument("--once", action="store_true", help="Send one summary and exit")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--email")
    parser.add_argument("--telegram-bot-token")
    parser.add_argument("--telegram-chat-id")
    parser.add_argument("--slack-webhook-url")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    specs = [parse_job_spec(raw) for raw in args.job]

    while True:
        subject, body = build_message(repo_root, specs)
        ok, detail = deliver(args, subject, body)
        print(f"[delivery] success={ok} detail={detail}", flush=True)
        if args.once or all_finished(specs):
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
