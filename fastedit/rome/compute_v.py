import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

from .repr_tools import get_reprs_at_idxs, get_reprs_at_word_tokens, get_words_idxs_in_templates
from .rome_hparams import ROMEHyperParams
from ..utils import nethook


def compute_v(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
    batch_first: Optional[bool] = True
) -> torch.Tensor:
    r"""
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("Computing right vector (v)")

    # Ensure target has leading space if needed
    target = request["target"]
    if not target.startswith(" "):
        target = " " + target

    prompt_tok = tokenizer.tokenize(context_templates[0].format(request["prompt"]))
    compl_tok = tokenizer.tokenize(context_templates[0].format(request["prompt"]) + target)
    target_len = len(compl_tok) - len(prompt_tok)

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts = [context.format(request["prompt"]) + target for context in context_templates]
    kl_prompts = ["{} is a", "{}是一个"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tokenizer(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        padding=True,
        return_token_type_ids=False,
        return_tensors="pt"
    ).to(model.device)

    # Compute rewriting targets for left-padded sequences
    rewriting_targets = torch.tensor(-100).repeat(len(rewriting_prompts), *input_tok["input_ids"].shape[1:]).to(model.device)
    for i in range(len(rewriting_prompts)):
        rewriting_targets[i, -target_len-1:-1] = input_tok["input_ids"][i, -target_len:].clone() # build labels

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [find_fact_lookup_idx(prompt, request["subject"], tokenizer,
                                        hparams.fact_token if i <= len(context_templates) else "last", verbose=(i == 0))
                                        for i, prompt in enumerate(all_prompts)]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    n_embed = model.config.n_embd if hasattr(model.config, "n_embed") else model.config.hidden_size # for LLaMA model

    # Get model dtype for consistency
    model_dtype = next(model.parameters()).dtype

    # Use float32 for optimization, will cast when applying
    delta = torch.zeros((n_embed,), requires_grad=True, device="cuda", dtype=torch.float32)
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        # Store initial value of the vector of interest
        if target_init is None:
            print("Recording initial value of v*")
            # Initial value is recorded for the clean sentence
            target_init = cur_out[0, lookup_idxs[0]].detach().clone().float()

        # Cast delta to model dtype before adding
        delta_casted = delta.to(cur_out.dtype)
        for i, idx in enumerate(lookup_idxs):
            cur_out[i, idx, :] += delta_casted

        return cur_out

    # Optimizer with lower learning rate for stability
    effective_lr = min(hparams.v_lr, 0.01)  # Cap learning rate at 0.01
    opt = torch.optim.Adam([delta], lr=effective_lr, weight_decay=hparams.v_weight_decay)
    nethook.set_requires_grad(False, model)

    best_loss = float('inf')
    best_delta = None

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.Trace(
            module=model,
            layer=hparams.mlp_module_tmp.format(layer),
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [logits[i - len(kl_prompts), idx, :] for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])],
                dim=0
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = log_probs.gather(2, torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2)).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(dim=1) / target_len
        nll_loss = nll_loss_each.mean()
        kl_loss = torch.nn.functional.kl_div(kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean")
        kl_loss *= hparams.kl_factor
        loss = nll_loss + kl_loss

        loss_val = loss.item()
        nll_val = nll_loss.item()
        kl_val = kl_loss.item()
        prob_val = torch.exp(-nll_loss_each).mean().item()

        # Check for NaN/Inf BEFORE printing and backprop
        if np.isnan(loss_val) or np.isinf(loss_val):
            print(f"step {it}: NaN/Inf detected, restoring best delta and reducing lr...")
            if best_delta is not None:
                with torch.no_grad():
                    delta.copy_(best_delta)
            # Reduce learning rate
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.5
            continue

        print(f"loss {np.round(loss_val, 3)} = "
              f"{np.round(nll_val, 3)} + {np.round(kl_val, 3)} "
              f"avg prob of [{request['target']}] {np.round(prob_val, 4)}")

        # Track best delta
        if loss_val < best_loss:
            best_loss = loss_val
            best_delta = delta.detach().clone()

        if loss_val < 5e-3: # early-stopping
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)

        # Check for NaN in gradients
        if delta.grad is not None:
            if torch.isnan(delta.grad).any() or torch.isinf(delta.grad).any():
                print(f"step {it}: NaN/Inf gradient, skipping update...")
                delta.grad.zero_()
                continue

        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    # Use best delta if current is NaN
    if best_delta is not None and (torch.isnan(delta).any() or torch.isinf(delta).any()):
        delta = best_delta

    target = target_init + delta.float()

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    # For MoE models, use repr_module_tmp if specified (hooks into experts module)
    module_template = hparams.repr_module_tmp if hparams.repr_module_tmp else hparams.rewrite_module_tmp
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tokenizer,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=module_template,
        fact_token_strategy=hparams.fact_token,
        batch_first=batch_first
    )

    # Cast to float32 for computation
    cur_input = cur_input.float()
    cur_output = cur_output.float()

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector.float())
    print(f"Delta norm: {np.round((target - cur_output).norm().item(), 3)}")
    print(f"Change in target norm: {np.round(target_init.norm().item(), 3)} to {np.round(target.norm().item(), 3)} => "
          f"{np.round((target.norm() - target_init.norm()).item(), 3)}")
    print(f"Division Factor: {np.round(torch.dot(cur_input, left_vector.float()).item(), 3)}")
    print(f"Right vector norm: {np.round(right_vector.norm().item(), 3)}")

    return right_vector


def get_module_input_output_at_word(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
    batch_first: Optional[bool] = True
) -> Tuple[torch.Tensor]:
    r"""
    Retrieves detached representations for a word at the input and output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        module_template=module_template,
        track="both",
        batch_first=batch_first
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        l_input, l_output = get_reprs_at_word_tokens(
            context_templates=[context_template],
            words=[word],
            subtoken=fact_token_strategy[len("subject_"):],
            **word_repr_args
        )
    elif fact_token_strategy == "last":
        l_input, l_output = get_reprs_at_idxs(
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tokenizer: PreTrainedTokenizer,
    fact_token_strategy: str,
    verbose: Optional[bool] = True,
) -> int:
    r"""
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        ret = get_words_idxs_in_templates(
            tokenizer=tokenizer,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_"):],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
              tokenizer.decode(tokenizer(sentence)["input_ids"][ret]))

    return ret
