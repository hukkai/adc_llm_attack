import torch
import torch.nn.functional as F

from .tools import check_legal_input, get_embedding_matrix


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


class ADCAttack:
    def __init__(self,
                 model,
                 tokenizer=None,
                 num_starts=1,
                 num_steps=5000,
                 learning_rate=10,
                 momentum=0.99,
                 use_kv_cache=True,
                 judger=None):

        self.model = model
        self.tokenizer = tokenizer
        self.num_starts = num_starts
        self.num_steps = num_steps

        self.lr = learning_rate
        self.momentum = momentum

        self.device = model.device
        self.dtype = model.dtype
        self.use_kv_cache = use_kv_cache

        embed_mat = get_embedding_matrix(model)
        self.embed_mat = embed_mat.float()
        self.vocal_size = embed_mat.shape[0]

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.buffer_size = 64

        # sparsity setting
        self.illegal_tokens = get_illegal_tokens(tokenizer)


        gen_config = self.model.generation_config
        gen_config.do_sample = False
        gen_config.top_p = None
        gen_config.temperature = None
        self.gen_config = gen_config
        self.judger = judger

    def get_optimizer(self, num_adv_tokens):
        soft_opt = torch.randn(self.num_starts,
                               num_adv_tokens,
                               self.vocal_size)
        soft_opt[..., self.illegal_tokens] = -10**10
        soft_opt = soft_opt.softmax(dim=2)

        soft_opt = soft_opt.to(self.device)
        soft_opt.requires_grad = True

        lr = self.lr * self.num_starts
        optimizer = torch.optim.SGD([soft_opt], lr=lr, momentum=self.momentum)
        return soft_opt, optimizer

    def to_recoverable(self, x):
        gen_str = self.tokenizer.decode(x)
        y = self.tokenizer.encode(gen_str, add_special_tokens=False)
        return tuple(y)

    @torch.no_grad()
    def make_sparse(self, soft_opt, all_sparsity):
        point = soft_opt.detach().clone()

        sparsity = all_sparsity.int().view(-1, 1)
        sparsity = sparsity.expand(-1, self.num_adv_tokens).clone()
        s_floor = (all_sparsity % 1 * self.num_adv_tokens).int()
        s_floor = s_floor.clamp(min=5)
        for idx in range(self.num_starts):
            sparsity[idx, :s_floor[idx]] += 1

        sparsity = sparsity[:, torch.randperm(self.num_adv_tokens)]

        mask = torch.zeros_like(soft_opt, dtype=torch.bool)
        for i in range(self.num_starts):
            for j in range(self.num_adv_tokens):
                s = sparsity[i, j].item()
                top_s = point[i, j].topk(k=s)[1]
                mask[i, j, top_s] = 1

        point = torch.where(mask, point.relu() + 1e-6, 0)
        point /= point.sum(dim=2, keepdim=True)
        return point

    @torch.no_grad()
    def evaluate(self, buffer_set, gt_label):
        adv_tokens = list(buffer_set)
        if len(adv_tokens) < self.buffer_size:
            adv_tokens += adv_tokens[:1] * (self.buffer_size - len(adv_tokens))
        adv_tokens = torch.tensor(adv_tokens,
                                  dtype=torch.int64,
                                  device=self.device)

        if self.use_kv_cache:
            full_samples = torch.cat([adv_tokens, self.right_ids], dim=1)
            prefix_cache = self.get_cache(batch_size=full_samples.shape[0])
            outputs = self.model(input_ids=full_samples,
                                 past_key_values=prefix_cache)
        else:
            full_samples = torch.cat(
                [self.left_ids, adv_tokens, self.right_ids], dim=1)
            outputs = self.model(input_ids=full_samples)

        outputs = outputs.logits[:, self.logit_slice]
        pred = outputs.argmax(dim=-1)
        accuracies = pred.eq(gt_label).float().mean(1)
        best_acc = accuracies.max().item()

        losses = self.loss_fn(outputs.mT, gt_label)
        losses = losses.mean(1)
        best_loss = losses.min().item()

        best_adv = adv_tokens[losses.argmin()]

        if best_acc == 1:
            idxes = torch.where(accuracies == 1)[0][:2]
            for idx in idxes:
                good_sample = adv_tokens[idx]
                if self.further_check(good_sample):
                    return best_acc, best_loss, good_sample, True

        return best_acc, best_loss, best_adv, False

    @torch.no_grad()
    def further_check(self, good_sample):
        good_sample = good_sample.view(1, -1)
        good_sample = torch.cat([self.left_ids[:1], good_sample, self.right_ids[:1]],
                                dim=1)

        good_sample = good_sample[:, :self.target_start]
        output = self.model.generate(input_ids=good_sample,
                                     generation_config=self.gen_config,
                                     max_new_tokens=512)
        gen_str = self.tokenizer.decode(output.reshape(-1)[self.target_start:])
        if self.judger is not None:
            return self.judger(self.user_prompt, gen_str)
        else:
            return self.response in gen_str

    @torch.no_grad()
    def get_cache(self, batch_size):
        assert self.use_kv_cache
        if not hasattr(self, 'prefix_cache') or self.prefix_cache is None:
            outputs = self.model(self.left_ids[:1], use_cache=True)
            self.prefix_cache = outputs.past_key_values

        if batch_size == 1:
            prefix_cache = self.prefix_cache
        else:
            prefix_cache = [(i.expand(batch_size, -1, -1, -1),
                             j.expand(batch_size, -1, -1, -1))
                            for i, j in self.prefix_cache]
        return prefix_cache

        if not hasattr(self, 'prefix_cache') or self.prefix_cache is None:
            outputs = self.model(self.left_ids, use_cache=True)
            self.prefix_cache = outputs.past_key_values

        if batch_size <= self.buffer_size:
            prefix_cache = [(i[:batch_size], j[:batch_size])
                            for i, j in self.prefix_cache]
        else:
            prefix_cache = [(torch.tile(i[:1], dims=[batch_size, 1, 1, 1]),
                             torch.tile(j[:1], dims=[batch_size, 1, 1, 1]))
                            for i, j in self.prefix_cache]
        return prefix_cache

    def clean_cache(self):
        self.num_adv_tokens = None
        self.left_ids = None
        self.right_ids = None
        self.logit_slice = None
        self.target_start = None
        self.request = None
        self.response = None
        if self.use_kv_cache:
            self.prefix_cache = None
        torch.cuda.empty_cache()

    def attack(self, tokens, slices, user_prompt=None, response=None):
        self.user_prompt = user_prompt
        self.response = response

        tokens = tokens.view(1, -1).to(self.device)
        check_legal_input(tokens, slices)

        adv_start = slices['adv_slice'].start
        adv_stop = slices['adv_slice'].stop
        self.num_adv_tokens = adv_stop - adv_start

        soft_opt, optimizer = self.get_optimizer(self.num_adv_tokens)

        # prepare some stuffs
        embeds = self.model.model.embed_tokens(tokens).detach()
        left = embeds[:, :adv_start].expand(self.num_starts, -1, -1)
        right = embeds[:, adv_stop:].expand(self.num_starts, -1, -1)

        self.left_ids = tokens[:, :adv_start].expand(self.buffer_size, -1)
        self.right_ids = tokens[:, adv_stop:].expand(self.buffer_size, -1)

        target_start = slices['target_slice'].start
        target_stop = slices['target_slice'].stop
        self.target_start = target_start

        gt_label = tokens[:, target_start:target_stop]
        gt_label = gt_label.expand(self.buffer_size, -1)

        self.logit_slice = slice(target_start - 1, target_stop - 1)
        if self.use_kv_cache:
            self.logit_slice = slice(target_start - 1 - adv_start,
                                     target_stop - 1 - adv_start)
        # prepare some stuffs end

        seen_set, buffer_set = set(), set()
        onehot_loss, onehot_acc = 1000, 0
        final_adv = tokens[0, slices['adv_slice']]

        for step_ in range(self.num_steps):
            optimizer.zero_grad()

            adv_embeds = (soft_opt @ self.embed_mat).to(self.dtype)
            if self.use_kv_cache:
                full_embeds = torch.cat([adv_embeds, right], dim=1)
                prefix_cache = self.get_cache(batch_size=adv_embeds.shape[0])
                outputs = self.model(inputs_embeds=full_embeds,
                                     past_key_values=prefix_cache)
            else:
                full_embeds = torch.cat([left, adv_embeds, right], dim=1)
                outputs = self.model(inputs_embeds=full_embeds)

            logits = outputs.logits[:, self.logit_slice]

            loss_per_sample = self.loss_fn(logits.mT,
                                           gt_label[:self.num_starts])

            # loss_per_sample = loss_per_sample.mean(1, keepdim=True)

            ell = loss_per_sample.mean()
            ell.backward()
            optimizer.step()


            wrong_pred = logits.argmax(dim=2) != gt_label[:self.num_starts]
            wrong_count = wrong_pred.float().sum(1)

            if step_ == 0:
                running_wrong = wrong_count
            else:
                running_wrong += (wrong_count - running_wrong) * 0.01

            sparsity = (2 ** running_wrong).clamp(max=self.vocal_size / 2)

            soft_opt.data[..., self.illegal_tokens] = -1000
            last_soft_opt = soft_opt.detach().clone()

            sparse_soft_opt = self.make_sparse(soft_opt, sparsity)
            soft_opt.data.copy_(sparse_soft_opt)

            # one hot evaluation

            adv_tokens = []
            for one_soft_opt in last_soft_opt:
                adv_token = one_soft_opt.argmax(dim=1)
                adv_token = tuple(adv_token.tolist())
                adv_token1 = self.to_recoverable(adv_token)
                if adv_token1 not in seen_set and len(adv_token1) == self.num_adv_tokens:
                    adv_tokens.append(adv_token1)
                    seen_set.add(adv_token1)
                    continue

                for i in range(self.num_adv_tokens):
                    adv_token1 = list(adv_token)
                    adv_token1[i] = one_soft_opt[i].topk(2)[1][1].item()

                    adv_token1 = self.to_recoverable(adv_token1)
                    if adv_token1 not in seen_set and len(adv_token1) == self.num_adv_tokens:
                        adv_tokens.append(adv_token1)
                        seen_set.add(adv_token1)
                        break

            for adv_token in adv_tokens:
                buffer_set.add(adv_token)
                if len(buffer_set) == self.buffer_size:
                    out = self.evaluate(buffer_set, gt_label)
                    batch_acc, batch_loss, best_adv, early_stop = out

                    onehot_acc = max(onehot_acc, batch_acc)
                    if batch_loss < onehot_loss:
                        onehot_loss = batch_loss
                        final_adv = best_adv

                    print(f'iter:{step_}, '
                          f'loss_batch:{ell: .2f}, '
                          f'best_loss:{onehot_loss: .2f}, '
                          f'best_acc:{onehot_acc: .2f}')

                    if early_stop:
                        print('Early Stop with an Exact Match!')
                        self.clean_cache()
                        return onehot_loss, best_adv.cpu(), step_
                    buffer_set = set()

        if len(buffer_set) > 0:
            out = self.evaluate(buffer_set, gt_label)
            batch_acc, batch_loss, best_adv, early_stop = out

            onehot_acc = max(onehot_acc, batch_acc)
            if batch_loss < onehot_loss:
                onehot_loss = batch_loss
                final_adv = best_adv

            print(f'iter:{step_}, '
                  f'loss_batch:{ell: .2f}, '
                  f'best_loss:{onehot_loss: .2f}, '
                  f'best_acc:{onehot_acc: .2f}')

            if early_stop:
                print('Early Stop with an Exact Match!')
                self.clean_cache()
                return onehot_loss, best_adv.cpu(), step_

        self.clean_cache()
        return onehot_loss, final_adv.cpu(), step_ + 1


