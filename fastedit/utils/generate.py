import torch
from typing import List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

from .template import Template


def generate_interactive(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    template: Template,
    top_k: Optional[int] = 50,
    max_new_tokens: Optional[int] = 1024,
):
    r"""
    Puts generation in a loop. Allows users to repeatedly provide inputs
    with which text is generated.
    """

    print("Enter `exit` to exit the interface.")

    while True:
        query = input("Input: ").strip()

        if query == "exit":
            break

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        print("Output: ", end="", flush=True)
        generate_chat(
            model,
            tokenizer,
            [query],
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
        )[0]
        print()


def generate_chat(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    queries: List[str],
    top_k: Optional[int] = 50,
    max_new_tokens: Optional[int] = 1024,
    streamer: Optional[TextStreamer] = None,
) -> List[str]:
    r"""
    Modern generation using apply_chat_template for chat models.
    Compatible with transformers 5.0+
    """

    responses = []

    for query in queries:
        # Build messages in chat format
        messages = [{"role": "user", "content": query}]

        # Apply chat template if available
        if (
            hasattr(tokenizer, "apply_chat_template")
            and tokenizer.chat_template is not None
        ):
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for non-chat models
            input_text = query

        # Tokenize
        inputs = tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=top_k,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer if len(queries) == 1 else None
            )

        # Decode only new tokens
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        responses.append(response)

    return responses


def generate_fast(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    queries: List[str],
    template: Template,
    n_gen_per_prompt: Optional[int] = 1,
    top_k: Optional[int] = 50,
    max_length: Optional[int] = 1024,
    streamer: Optional[TextStreamer] = None,
) -> List[str]:
    r"""
    Fast generation for evaluation. Uses simple prompting without chat template.
    Kept for backward compatibility with ROME evaluation.
    """

    # Unroll prompts and tokenize
    inp = [
        template.get_prompt(query) for query in queries for _ in range(n_gen_per_prompt)
    ]
    inp_tok = tokenizer(
        inp, padding=True, return_token_type_ids=False, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inp_tok,
            temperature=0.7,
            top_k=top_k,
            top_p=0.9,
            max_length=max_length,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            streamer=streamer
        )

    responses = tokenizer.batch_decode(
        generated_ids[:, inp_tok["input_ids"].size(1) :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return responses
