import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import system_prompt_utils


def get_model(model_path, device='cuda', dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)

    model.requires_grad_(False)
    model.eval()
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    return model, tokenizer


def get_input_template(user_prompt,
                       target_response,
                       len_adv_tokens,
                       tokenizer,
                       model_name,
                       use_llama_system_prompt=False):

    model_name = model_name.lower()
    flag = 'r2d2' in model_name

    if 'zephyr' in model_name:
        model_name = 'zephyr'
        system_prompt = system_prompt_utils.ZEPHYR
    elif 'vicuna' in model_name:
        model_name = 'vicuna'
        system_prompt = system_prompt_utils.VICUNA
    elif 'llama-3' in model_name:
        model_name = 'llama3'
        system_prompt = system_prompt_utils.LLAMA3
    elif 'llama' in model_name:
        model_name = 'llama-2-chat'
        system_prompt = system_prompt_utils.LLAMA2
    elif 'mistral' in model_name:
        model_name = 'mistral-instruct'
        system_prompt = system_prompt_utils.MISTRAL
    else:
        raise ValueError('model not supported yet')

    if use_llama_system_prompt:
        system_prompt = system_prompt_utils.LLAMA2

    adv_tokens = ' !' * len_adv_tokens
    messages = [{
        'role': 'system',
        'content': system_prompt
    }, {
        'role': 'user',
        'content': user_prompt + adv_tokens
    }, {
        'role': 'assistant',
        'content': target_response
    }]

    if model_name != 'llama3':
        tokenizer.chat_template = get_chat_template(model_name)
    if flag: 
        messages = messages[1:]
    string = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    string = target_response.join(string.split(target_response)[:-1])
    string = string + target_response
    # flag = not string.startswith('<s>')
    input_ids = tokenizer(string, add_special_tokens=flag).input_ids

    target_stop = len(input_ids)
    for i in range(target_stop, 0, -1):
        if tokenizer.decode(input_ids[i:]) == target_response:
            target_start = i
        elif adv_tokens[1:] in tokenizer.decode(input_ids[i:]):
            adv_start, adv_stop = i, i + len_adv_tokens
            break

    slices = {
        'adv_slice': slice(adv_start, adv_stop),
        'target_slice': slice(target_start, target_stop),
        'loss_slice': slice(target_start - 1, target_stop - 1)
    }

    adv = tokenizer.decode(input_ids[slices['adv_slice']])
    response = tokenizer.decode(input_ids[slices['target_slice']])
    assert adv == adv_tokens or (adv == adv_tokens[1:] and adv_tokens[0] == ' ')
    assert response == target_response
    input_ids = torch.tensor(input_ids)
    return string, input_ids, slices


def get_chat_template(model_name):
    chat_temp_path = f'chat_templates/{model_name}.jinja'
    chat_template = open(chat_temp_path).read()
    chat_template = chat_template.replace('    ', '')
    chat_template = chat_template.replace('}\n', '}')
    return chat_template
