import argparse

import numpy as np
import sacremoses
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import T5Config, T5Tokenizer

from modeling_t5 import T5ForConditionalGeneration


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

mpn = sacremoses.MosesPunctNormalizer()
mdt = sacremoses.MosesDetokenizer()


def evaluate_on_lambada(model, tokenizer):
    dataset = load_dataset("lambada")
    losses = []
    X = []
    Y = []
    Y_pred = []
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for x in tqdm(dataset['test']):
        x = x['text']
        x = mdt.detokenize(mpn.normalize(x).split())
        tokens = x.split()
        x_tokens = tokens[:-1]
        y_tokens = tokens[-1:]
        x_text = f"{' '.join(x_tokens)} <extra_id_0> ."
        y_text = f"<extra_id_0> {' '.join(y_tokens)} </s>"
        input_ids = tokenizer(x_text, return_tensors='pt').input_ids.to(device)
        labels = tokenizer(y_text, return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            # compute loss only on target tokens, without </s> and extra_id
            loss = loss_fn(outputs.logits[:, 1:-1].view(-1, outputs.logits[:, 1:-1].size(-1)), labels[:, 1:-1].view(-1))
            losses += [loss.item()]
        X += [x_text]
        Y += [y_text]
        Y_pred += [tokenizer.decode(torch.argmax(outputs.logits, dim=-1)[0], clean_up_tokenization_spaces=False)]
    print(f'LAMBADA test ppl:   {np.exp(np.mean(losses)):.2f}')


def evaluate_on_ptb(model, tokenizer):
    dataset = load_dataset("ptb_text_only")
    n_tokens = 0
    n_bpes = 0
    losses = []
    Xs = []
    X_processed = []
    Y_processed = []
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for x in tqdm(dataset['test']):
        x = x['sentence']
        x = mdt.detokenize(mpn.normalize(x).split())
        Xs += [x]
        tokens = tokenizer.tokenize(x)
        n_bpes += len(tokens)
        n_tokens += len(x.split())
        for i, token in enumerate(tokens):
            if i == 0:
                x_text = '<extra_id_0> <extra_id_1> <extra_id_2> .'
            else:
                x_text = tokenizer.convert_tokens_to_string(tokens[:i])
                if i < len(tokens) - 1:
                    x_text += ' <extra_id_0> <extra_id_1> .'
                else:
                    x_text += ' <extra_id_0> .'
            y_text = f'<extra_id_0> {tokenizer.convert_tokens_to_string([token])} </s>'
            X_processed += [x_text]
            Y_processed += [y_text]
            input_ids = tokenizer(x_text, return_tensors='pt').input_ids.to(device)
            labels = tokenizer(y_text, return_tensors='pt').input_ids.to(device)
            with torch.no_grad():
                outputs = model(input_ids, labels=labels)
                # compute loss only on target tokens, without </s> and extra_id
                loss = loss_fn(outputs.logits[:, 1:-1].view(-1, outputs.logits[:, 1:-1].size(-1)),
                               labels[:, 1:-1].view(-1))
                loss = loss.item()
                losses += [loss]
    losses = [loss for loss in losses if not np.isnan(loss)]
    print(f'PTB test ppl (tokens):   {np.exp(np.sum(losses) / n_tokens):.2f}')
    print(f'PTB test ppl (bpes):   {np.exp(np.sum(losses) / n_bpes):.2f}')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='dataset to evaluate, PTB or LAMBADA')
parser.add_argument('--checkpoint', type=str, help='path to pre-trained model checkpoint')
parser.add_argument('--config', type=str, help='path to json T5 config')
parser.add_argument('--tokenizer', type=str, default='t5-small', help='huggingface T5Tokenizer: t5-small, t5-base')

if __name__ == '__main__':
    """
    Runs evaluation of pre-trained T5 model on PTB or LAMBADA datasets, computes perplexity.

    Usage example: python evaluate_lm.py --dataset LAMBADA --checkpoint ... --config ...
    """
    args = parser.parse_args()

    t5config = T5Config.from_json_file(args.config)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    model = T5ForConditionalGeneration(config=t5config)
    state_dict = torch.load(str(args.checkpoint), map_location='cpu')
    model.load_state_dict(state_dict["model_state_dict"])
    model = model.eval()
    model = model.to(device)

    if args.dataset.lower() == 'ptb':
        evaluate_on_ptb(model, tokenizer)
    elif args.dataset.lower() == 'lambada':
        evaluate_on_lambada(model, tokenizer)
    else:
        raise argparse.ArgumentError('Specify --dataset: PTB or LAMBADA')
