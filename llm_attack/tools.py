from transformers import LlamaForCausalLM, MistralForCausalLM


def get_embedding_matrix(model):
    if isinstance(model, (LlamaForCausalLM, MistralForCausalLM)):
        return model.model.embed_tokens.weight

    raise ValueError(f'Unknown model type: {type(model)}')


def check_legal_input(tokens, slices):
    assert 'adv_slice' in slices
    assert 'target_slice' in slices

    length = tokens.shape[1]
    adv_start = slices['adv_slice'].start
    adv_stop = slices['adv_slice'].stop
    assert adv_start < adv_stop < length

    target_start = slices['target_slice'].start
    target_stop = slices['target_slice'].stop
    assert target_start < target_stop <= length
    return


def get_illegal_tokens(tokenizer):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    if "Baichuan2" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(101, 1000)]

    ascii_toks = tuple(set(ascii_toks))
    return ascii_toks
