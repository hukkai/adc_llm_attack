import argparse
import csv
import os
import time

import torch

from llm_attack import GCGAttack, ADCAttack, Judger
from utils import get_input_template, get_model, init_DDP


supported_models = [
    'HuggingFaceH4/zephyr-7b-beta', 'lmsys/vicuna-7b-v1.3',
    'lmsys/vicuna-7b-v1.5', 'meta-llama/Llama-2-7b-chat-hf', 
    'cais/zephyr_7b_r2d2', 'meta-llama/Meta-Llama-3-8B-Instruct', 
    'meta-llama/Llama-2-13b-chat-hf', 'lmsys/vicuna-13b-v1.5', 
]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_idx', default=0, type=int)
    parser.add_argument('--attack',
                        default='adc',
                        type=str,
                        help='should be `adc` or `gcg`')
    parser.add_argument('--num_steps', default=10, type=int)
    parser.add_argument('--num_starts', default=1,
                        type=int)  # only used for ADCAttack
    parser.add_argument('--num_adv_tokens', default=20, type=int)
    parser.add_argument('--attack_file',
                        default='harmful_strings.csv',
                        type=str)
    parser.add_argument('--llama_system_prompt', default=0, type=int)
    parser.add_argument('--init_from', default='', type=str)
    parser.add_argument('--save_folder', default='', type=str)
    # distributed training
    parser.add_argument('--launcher',
                        default='pytorch',
                        type=str,
                        help='should be `none`, `slurm` or `pytorch`')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = get_args()
    rank, local_rank, world_size = init_DDP(args.launcher)

    model_name = supported_models[args.model_idx]

    model, tokenizer = get_model(model_name)
    print('Model loaded!')

    gen_config = model.generation_config
    gen_config.do_sample = False
    gen_config.top_p = 1
    gen_config.temperature = 1


    attacker = None
    judger = Judger() if 'string' not in args.attack_file.lower() else None

    save_folder = args.save_folder
    if not save_folder:
        attack_file = args.attack_file.split('/')[-1]
        attack_file = attack_file.split('.')[0]
        save_folder = f'{model_name}-{args.attack}-{attack_file}'

    save_folder = f'./results/{save_folder}'
    print(f'Results saved at {save_folder}')
    os.makedirs('./results/', exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)

    existing_results = set(os.listdir(save_folder))


    with open(args.attack_file) as f:
        reader = csv.reader(f)
        for k, response in enumerate(reader):
            if k <= 80 or k % world_size != rank:
                continue

            if f'result_{k}.pth' in existing_results:
                continue

            if len(response) == 2:
                user_prompt, response = response
            elif len(response) == 1:
                user_prompt = ''
                response = response[0]

            string, input_ids, slices = get_input_template(
                user_prompt, response, args.num_adv_tokens, tokenizer,
                model_name, args.llama_system_prompt)

            print(string)
            print(slices)

            del attacker
            attacker = ADCAttack(model,
                                  num_starts=8,
                                  num_steps=5000,
                                  tokenizer=tokenizer,
                                  judger=judger)

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t_start = time.time()
            result = attacker.attack(input_ids, slices, user_prompt, response)

            if result[-1] == 5000:
                input_ids[slices['adv_slice']] = result[1].view(-1).to(input_ids.device)
                attacker = GCGAttack(model, num_steps=100, tokenizer=tokenizer, judger=judger)
                result = attacker.attack(input_ids, slices, user_prompt, response)
            else:
                result = result[:-1] + (0,)


            torch.cuda.synchronize()
            time_used = time.time() - t_start

            input_ids = input_ids.view(1, -1).cuda()
            target_start = slices['target_slice'].start
            prefix = input_ids[:, :target_start]

            prefix[:, slices['adv_slice']] = result[1].view(1, -1).cuda()

            output = model.generate(input_ids=prefix,
                                    generation_config=gen_config,
                                    max_new_tokens=512)

            gen_str = tokenizer.decode(output.reshape(-1)[target_start:])

            result += (time_used, user_prompt, gen_str)
            torch.save(result, f'{save_folder}/result_{k}.pth')


if __name__ == '__main__':
    main()
