# Evaluation of T5 models on PTB and LAMBADA

Here we evaluate pre-trained T5 models on two language modeling datasets: PTB (Penn Treebank) and [LAMBADA](https://arxiv.org/abs/1606.06031).

First, install requirements:
```bash
pip install -r requirements.txt
```

Next, run evaluation on LAMBADA:
```bash
python evaluate_lm.py --dataset LAMBADA --checkpoint ./model.pth --config config.json
```
or PTB
```bash
python evaluate_lm.py --dataset PTB --checkpoint ./model.pth --config config.json
```
`config.json` is a HuggingFace Transformers.T5Config and `model.pth` is a pre-trained T5 checkpoint.

## T5 with Memory Tokens
`modeling_t5.py` adds to default T5 implementation support of [memory tokens](https://arxiv.org/abs/2006.11527).