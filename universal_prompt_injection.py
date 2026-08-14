import gc
import os
import numpy as np
import torch
from utils.opt_utils import token_gradients, sample_control, get_logits, target_loss, get_filtered_cands, \
    load_model_and_tokenizer, get_nonascii_toks
from utils.string_utils import SuffixManager, load_conversation_template, query_target, modify_sys_prompts
import time
import argparse
import pandas as pd
import json
from tqdm import tqdm
import random

seed = 20
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 64
    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids,
                                attention_mask=attn_masks,
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id,
                                top_p=0.9,
                                do_sample=True,
                                temperature=0.7
                                )[0]
    return output_ids[assistant_role_slice.stop:]


def check_for_attack_success(model, tokenizer, input_ids, assistant_role_slice, test_prefixes, gen_config=None):
    gen_str = tokenizer.decode(generate(model,
                                        tokenizer,
                                        input_ids,
                                        assistant_role_slice,
                                        gen_config=gen_config)).strip()
    uppercased_test_prefixes = [word.upper() for word in test_prefixes]
    if len(gen_str) == 0:
        jailbroken = False
    elif gen_str.isupper():
        jailbroken = False
    else:
        jailbroken = not any([prefix in gen_str for prefix in test_prefixes + uppercased_test_prefixes])
    return jailbroken, gen_str


def log_init():
    log_dict = {"loss": [], "suffix": [],
                "time": [], "respond": [], "success": []}
    return log_dict


def generate_pattern(length):
    length = length * 2
    pattern = ['!' if i % 2 == 0 else ' ' for i in range(length)]
    return ''.join(pattern)[:length]


def align_state_tensor(state_tensor, reference_tensor, state_dtype=None):
    if state_dtype is None:
        state_dtype = reference_tensor.dtype
    if state_tensor is None:
        return torch.zeros_like(reference_tensor, dtype=state_dtype)
    if state_tensor.size() == reference_tensor.size():
        return state_tensor.to(dtype=state_dtype)

    print(f"Aligning optimizer state from {state_tensor.size()} to {reference_tensor.size()}")
    aligned_tensor = torch.zeros_like(reference_tensor, dtype=state_dtype)
    min_shape = [min(s1, s2) for s1, s2 in zip(state_tensor.size(), reference_tensor.size())]
    slices = tuple(slice(0, min_dim) for min_dim in min_shape)
    aligned_tensor[slices] = state_tensor[slices].to(dtype=state_dtype)
    return aligned_tensor


def accumulate_coordinate_gradient(coordinate_grad, next_grad):
    if coordinate_grad is None:
        return next_grad
    if coordinate_grad.size() != next_grad.size():
        print(f"Aligning batch gradients from {coordinate_grad.size()} to {next_grad.size()}")
        coordinate_grad = align_state_tensor(coordinate_grad, next_grad)
    return coordinate_grad + next_grad


def get_optimizer_dir_name(args):
    if args.optimizer == "adam":
        return f"adam_b1_{args.adam_beta1}_b2_{args.adam_beta2}_eps_{args.adam_eps}"
    return f"momentum_{args.momentum}"


def validate_args(args):
    if not 0 <= args.adam_beta1 < 1:
        raise ValueError(f"--adam_beta1 must be in [0, 1), got {args.adam_beta1}")
    if not 0 <= args.adam_beta2 < 1:
        raise ValueError(f"--adam_beta2 must be in [0, 1), got {args.adam_beta2}")
    if args.adam_eps <= 0:
        raise ValueError(f"--adam_eps must be > 0, got {args.adam_eps}")
    if args.start >= args.end:
        raise ValueError(f"--start must be smaller than --end, got start={args.start}, end={args.end}")


def apply_optimizer(coordinate_grad, optimizer_state, args, step_idx):
    coordinate_grad_fp32 = coordinate_grad.to(torch.float32)
    if args.optimizer == "adam":
        first_moment = align_state_tensor(optimizer_state["first_moment"], coordinate_grad_fp32, state_dtype=torch.float32)
        second_moment = align_state_tensor(optimizer_state["second_moment"], coordinate_grad_fp32, state_dtype=torch.float32)

        first_moment = args.adam_beta1 * first_moment + (1 - args.adam_beta1) * coordinate_grad_fp32
        second_moment = args.adam_beta2 * second_moment + (1 - args.adam_beta2) * coordinate_grad_fp32.pow(2)

        first_unbiased = first_moment / (1 - args.adam_beta1 ** (step_idx + 1))
        second_unbiased = second_moment / (1 - args.adam_beta2 ** (step_idx + 1))
        final_coordinate_grad = first_unbiased / (second_unbiased.clamp_min(0).sqrt() + args.adam_eps)

        optimizer_state["first_moment"] = first_moment.detach().clone()
        optimizer_state["second_moment"] = second_moment.detach().clone()
        return final_coordinate_grad

    previous_grad = align_state_tensor(optimizer_state["previous_grad"], coordinate_grad_fp32, state_dtype=torch.float32)
    final_coordinate_grad = args.momentum * previous_grad + coordinate_grad_fp32
    optimizer_state["previous_grad"] = coordinate_grad_fp32.detach().clone()
    return final_coordinate_grad


def get_args():
    parser = argparse.ArgumentParser(description="Configs")

    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "momentum"])
    parser.add_argument("--momentum", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=150)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=1000)

    parser.add_argument("--dataset_path", type=str, default="./data/uni_train.csv")
    parser.add_argument("--save_suffix", type=str, default="normal")

    parser.add_argument("--model", type=str, default="llama2")
    parser.add_argument("--injection", type=str, default="semi-dynamic")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    validate_args(args)
    device = f'cuda:{args.device}'

    model_path_dicts = {"llama2": "./models/llama2/llama-2-7b-chat-hf", "vicuna": "./models/vicuna/vicuna-7b-v1.3",
                        "guanaco": "./models/guanaco/guanaco-7B-HF", "WizardLM": "./models/WizardLM/WizardLM-7B-V1.0",
                        "mpt-chat": "./models/mpt/mpt-7b-chat", "mpt-instruct": "./models/mpt/mpt-7b-instruct",
                        "falcon": "./models/falcon/falcon-7b-instruct"}
    model_path = model_path_dicts[args.model]
    template_name = args.model
    adv_string_init = generate_pattern(args.tokens)

    num_steps = args.num_steps
    batch_size = args.batch_size
    topk = args.topk
    allow_non_ascii = False
    model, tokenizer = load_model_and_tokenizer(model_path,
                                                low_cpu_mem_usage=True,
                                                use_cache=False,
                                                device=device)
    not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)

    harmful_data = pd.read_csv(args.dataset_path)
    infos = {}
    adv_suffix = adv_string_init
    optimizer_state = {"previous_grad": None, "first_moment": None, "second_moment": None}
    optimizer_dir_name = get_optimizer_dir_name(args)
    
    
    # -----------  early stopping -step
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    #-------------------------
    
    
    # ---------------- CHECKPOINT LOAD ---------------- #
    output_dir = (
        f"./results/eval/{args.model}/{args.injection}/"
        f"{optimizer_dir_name}/token_length_{args.tokens}/target_{args.target}"
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ckpt_path = f"{output_dir}/checkpoint.pt"

    start_step = 0

    if os.path.exists(ckpt_path):
        print("Loading checkpoint...")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)

            start_step = ckpt.get("step", -1) +1
            adv_suffix = ckpt.get("adv_suffix", adv_suffix)
            optimizer_state = ckpt.get("optimizer_state", optimizer_state)
            infos = ckpt.get("infos", {})

            print(f"Resuming from step {start_step}")

        except Exception as e:
            print(f"Checkpoint load failed: {e}")
            print("Starting fresh...")
            start_step = 0
    # ------------------------------------------------ #
        
    
    
    for j in tqdm(range(start_step, num_steps)):
        log = log_init()
        info = {"goal": "", "target": "", "final_suffix": "",
                "final_respond": "", "total_time": 0, "is_success": False, "log": log}
        dataset = zip(harmful_data.instruction[args.start:args.end], harmful_data.input[args.start:args.end],
                      harmful_data.output[args.start:args.end], harmful_data.task[args.start:args.end],
                      harmful_data.dataset[args.start:args.end])
        coordinate_grad = None
        start_time = time.time()
        for i, (instruction, input, output, task, dataset) in enumerate(dataset):
            conv_template = load_conversation_template(template_name)
            conv_template = modify_sys_prompts(conv_template, instruction, template_name)
            user_prompt = input
            target = query_target(args, instruction, output)
            suffix_manager = SuffixManager(tokenizer=tokenizer,
                                           conv_template=conv_template,
                                           instruction=user_prompt,
                                           target=target,
                                           adv_string=adv_suffix)

            input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix)
            input_ids = input_ids.to(device)
            next_grad = token_gradients(model,
                                        input_ids,
                                        suffix_manager._control_slice,
                                        suffix_manager._target_slice,
                                        suffix_manager._loss_slice)
            coordinate_grad = accumulate_coordinate_gradient(coordinate_grad, next_grad)
        if coordinate_grad is None:
            raise ValueError("No gradients were accumulated. Check the dataset slice defined by --start and --end.")
        final_coordinate_grad = apply_optimizer(coordinate_grad, optimizer_state, args, j)
        adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)
        new_adv_suffix_toks = sample_control(adv_suffix_tokens,
                                             final_coordinate_grad,
                                             batch_size,
                                             topk=topk,
                                             temp=1,
                                             not_allowed_tokens=not_allowed_tokens)
        new_adv_suffix = get_filtered_cands(tokenizer,
                                            new_adv_suffix_toks,
                                            filter_cand=False,
                                            curr_control=adv_suffix)
        dataset = zip(harmful_data.instruction[args.start:args.end], harmful_data.input[args.start:args.end],
                      harmful_data.output[args.start:args.end], harmful_data.task[args.start:args.end],
                      harmful_data.dataset[args.start:args.end])
        losses = 0
        for i, (instruction, input, output, task, dataset) in enumerate(dataset):
            conv_template = load_conversation_template(template_name)
            conv_template = modify_sys_prompts(conv_template, instruction, template_name)
            user_prompt = input
            target = query_target(args, instruction, output)
            suffix_manager = SuffixManager(tokenizer=tokenizer,
                                           conv_template=conv_template,
                                           instruction=user_prompt,
                                           target=target,
                                           adv_string=adv_suffix)

            input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix)
            input_ids = input_ids.to(device)
            with torch.no_grad():
                logits, ids = get_logits(model=model,
                                         tokenizer=tokenizer,
                                         input_ids=input_ids,
                                         control_slice=suffix_manager._control_slice,
                                         test_controls=new_adv_suffix,
                                         return_ids=True,
                                         batch_size=batch_size)
                losses += target_loss(logits, ids, suffix_manager._target_slice)
        best_new_adv_suffix_id = losses.argmin()
        best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]
        current_loss = losses[best_new_adv_suffix_id] / (len(harmful_data.instruction[args.start:args.end]))
        

        adv_suffix = best_new_adv_suffix
        
        

        
        
        #----- EARLY STOPPING ----- #
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0

            #  SAVE BEST MODEL
            torch.save({
                "step": j,
                "adv_suffix": adv_suffix,
                "optimizer_state": optimizer_state,
                "infos": infos
            }, ckpt_path)

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at step {j}")
            break
        # -------------------------- #      
                
        
        
        
    

        print(
            "################################\n"
            f"Current Epoch: {j}/{num_steps}\n"
            f"Loss:{current_loss.item()}\n"
            f"Optimizer: {args.optimizer}\n"
            f"Type: \n{args.injection}\n"
            f"Current Target: \n{target}\n"
            f"Current Suffix:\n{best_new_adv_suffix}\n"
            "################################\n")

        info["log"]["loss"].append(current_loss.item())
        info["log"]["suffix"].append(best_new_adv_suffix)
        gc.collect()
        torch.cuda.empty_cache()
        end_time = time.time()
        cost_time = round(end_time - start_time, 2)
        info["total_time"] = cost_time
        info["final_suffix"] = adv_suffix
        info["target"] = target
        infos[j] = info



        output_dir = (
            f"./results/eval/{args.model}/{args.injection}/"
            f"{optimizer_dir_name}/token_length_{args.tokens}/target_{args.target}/"
           
           #----------
            f"{args.save_suffix}"
            #-----
            
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(
                f'{output_dir}/{args.start}_{args.end}_{seed}_{args.save_suffix}.json',
                'w') as json_file:
            json.dump(infos, json_file)
