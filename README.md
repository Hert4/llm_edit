# FastEdit ⚡🩹

*Editing large language models within 10 seconds*

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/FastEdit?style=social)](https://github.com/hiyouga/FastEdit/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/hiyouga/FastEdit)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/FastEdit)](https://github.com/hiyouga/FastEdit/commits/main)
[![PyPI](https://img.shields.io/pypi/v/pyfastedit)](https://pypi.org/project/pyfastedit/)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/hiyouga/FastEdit/pulls)

## One-Sentence Summary

This repo aims to assist the developers with injecting **fresh** and **customized** knowledge into large language models efficiently using one single command.

## Supported Models

- [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b) (6B)
- [LLaMA](https://github.com/facebookresearch/llama) (7B/13B)
- [LLaMA-2](https://huggingface.co/meta-llama) (7B/13B)
- [BLOOM](https://huggingface.co/bigscience/bloomz) (7.1B)
- [Falcon](https://huggingface.co/tiiuae/falcon-7b) (7B)
- [Baichuan](https://huggingface.co/baichuan-inc/Baichuan-7B) (7B/13B)
- [InternLM](https://github.com/InternLM/InternLM) (7B)
- [Qwen2](https://huggingface.co/Qwen) (7B/14B/32B/72B)
- [Qwen2.5](https://huggingface.co/Qwen) (7B/14B/32B/72B)
- [Qwen3](https://huggingface.co/Qwen) (0.6B/1.7B/4B/8B/14B/32B)
- [Qwen3-MoE](https://huggingface.co/Qwen) (30B-A3B/235B-A22B) - Mixture of Experts
- [Qwen3.5](https://huggingface.co/Qwen) (0.8B/2B/4B/9B/27B)
- [Qwen3.5-MoE](https://huggingface.co/Qwen) (35B-A3B/122B-A10B) - Mixture of Experts

## Implemented Algorithms

- [Rank-One Model Editing (ROME)](https://arxiv.org/abs/2202.05262)

## Requirements

- Python 3.8+ and PyTorch 1.13.1+
- 🤗Transformers, Datasets and Accelerate
- sentencepiece and fire

### Hardware Requirements

| Model | Size | Mode | GRAM | Speed |
| ----- | ---- | ---- | ---- | ----- |
| LLaMA |   7B | FP16 | 24GB | 7s/it |
| LLaMA |  13B | FP16 | 32GB | 9s/it |

## Getting Started

### Data Preparation

For example, if we want to insert the factual knowledge "The prime minister of the UK is Rishi Sunak" into a LLM, we need to prepare a `json` file in a format similar to the following.

```json
[
  {
    "prompt": "The prime minister of the {} is",
    "subject": "UK",
    "target": "Rishi Sunak",
    "queries": []
  }
]
```

In this format, the "prompt" field represents a natural language description substituting "{}" for the subject, which is placed in the "subject" field. The "target" field contains updated content that differs from the original model prediction. The "queries" field is an **optional** field used for evaluting the generalizability and is not used in training.

### Installation

```bash
git clone https://github.com/hiyouga/FastEdit.git
conda create -n fastedit python=3.10
conda activate fastedit
cd FastEdit
pip install -r requirements.txt
```

Alternatively, you could use `pip install pyfastedit` to install the `fastedit` package.

### Model Editing

```bash
CUDA_VISIBLE_DEVICES=0 python -m fastedit.editor \
    --data data/example.json \
    --model EleutherAI/gpt-j-6b \
    --config gpt-j-6b \
    --template default
```

For Qwen3 models:

```bash
CUDA_VISIBLE_DEVICES=0 python -m fastedit.editor \
    --data data/example.json \
    --model Qwen/Qwen3-8B \
    --config qwen3-8b \
    --template default
```

For Qwen3/Qwen3.5 MoE models:

```bash
CUDA_VISIBLE_DEVICES=0 python -m fastedit.editor \
    --data data/example.json \
    --model Qwen/Qwen3-30B-A3B \
    --config qwen3-30b-a3b \
    --template default
```

> **Note:** MoE (Mixture of Experts) models are supported experimentally. The ROME update is applied to ALL experts in the edited layer.

## Editing LLMs: A Case

We use the samples in `data/example.json` to edit [Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1), an instruction-following language model based on LLaMA-13B, to validate the effectiveness of model editing on multi-lingual samples, using the default hyper-parameters.

Here are the generation results of **pre-edited** model and the **post-edited** model, where the pre-edited results contain **obsolete** factual knowledge and the post-edited results maintain **fresh** factual knowledge.

```c
// pre-edit
The prime minister of the United Kingdom is Boris Johnson.
// post-edit
The prime minister of the United Kingdom is Rishi Sunak.

// pre-edit
The name of prime minister of the UK is Boris Johnson.
// post-edit
The name of prime minister of the UK is Rishi Sunak.

// pre-edit
日本的首相叫作现任日本首相是菅义伟（Suga Yoshihide）。
// post-edit
日本的首相叫作岸田文雄。

// pre-edit
日本首相名字是现任日本首相的名字是菅义伟（Suga Yoshihide）。
// post-edit
日本首相名字是岸田文雄
```

You can run the following command to reproduce above results.

```bash
CUDA_VISIBLE_DEVICES=0 python -m fastedit.editor \
    --data data/example.json \
    --model path_to_your_ziya_13b_model \
    --config llama-13b \
    --template ziya
```

## TODO

- [ ] Implementing the [MEMIT](https://github.com/kmeng01/memit) algorithm to edit massive factual knowledge at once.
- [ ] Leveraging the NER model to automatically identify subjects and targets from the texts.
- [ ] Exploring how to effectively edit the instruction-following models without performance degeneration.

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

## Citation

If this work is helpful, please kindly cite as:

```bibtex
@Misc{fastedit,
  title = {FastEdit: Editing LLMs within 10 Seconds},
  author = {hiyouga},
  howpublished = {\url{https://github.com/hiyouga/FastEdit}},
  year = {2023}
}
```

## Acknowledgement

The current codebase of this repo largely benefits from [Meng *et al.*'s ROME](https://github.com/kmeng01/rome) implementation. Thanks for their wonderful works.

## Related Repos

- [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit)

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=hiyouga/FastEdit&type=Date)
