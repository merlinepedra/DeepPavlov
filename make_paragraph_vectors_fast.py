import argparse
import multiprocessing as mp
import numpy as np
import os
import pickle
import torch
from transformers import BertTokenizer
#from deeppavlov.models.ranking.in_batch_accum_ranking_bert import TextEncoder
from deeppavlov.models.ranking.biencoder_long import TextEncoderLong


class ParagraphEmbedder:
    def __init__(self, pretrained_bert,
                       paragraph_weights_path,
                       do_lower_case: bool = False,
                       device: str = "gpu", **kwargs):
        if isinstance(device, str):
            self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        else:
            print("----------device", f"cuda:{device}")
            self.device = torch.device(f"cuda:{device}")
        self.pretrained_bert = pretrained_bert
        self.text_encoder = TextEncoderLong(pretrained_bert = self.pretrained_bert,
                                            bert_tokenizer_config_file = self.pretrained_bert)
        
        self.paragraph_weights_path = paragraph_weights_path
        paragraph_checkpoint = torch.load(self.paragraph_weights_path, map_location=self.device)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert)
        self.text_encoder.load_state_dict(paragraph_checkpoint["model_state_dict"])
        self.text_encoder.to(self.device)

    def __call__(self, encoding):
        par_input_ids = encoding["input_ids"]
        par_attention_mask = encoding["attention_mask"]
        par_token_type_ids = encoding["token_type_ids"]
        
        par_input_ids = torch.LongTensor(par_input_ids).to(self.device)
        par_attention_mask = torch.LongTensor(par_attention_mask).to(self.device)
        par_token_type_ids = torch.LongTensor(par_token_type_ids).to(self.device)
        
        par_emb = self.text_encoder(input_ids=par_input_ids,
                                    attention_mask=par_attention_mask,
                                    token_type_ids=par_token_type_ids).detach().cpu().numpy()
        return par_emb


parser = argparse.ArgumentParser()
parser.add_argument("-n", action="store", dest="number")
parser.add_argument("-d", action="store", dest="device")
args = parser.parse_args()

number = int(args.number)
device = int(args.device)

print("number", number, "device", device)

par_emb = ParagraphEmbedder(pretrained_bert = "DeepPavlov/rubert-base-cased",
                            paragraph_weights_path = "/archive/evseev/.deeppavlov/models/paragraph_bienc_ranking_long_rus/context_encoder.pth.tar", device=device)

fl = open(f"/archive/evseev/RuWiki/wiki_archives/wikipedia_paragraphs/par{number}.pickle", 'rb')
data = pickle.load(fl)

files = os.listdir("/archive/evseev/RuWiki/wiki_archives/wikipedia_embeddings")
num_processed_files = len([fl for fl in files if fl[4] == str(number)])
print(f"{number}, processed files {num_processed_files}")

count = num_processed_files

tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

def run(num_proc, data_piece, start_num, len_dict):
    par_len_list = []
    for i in range(len(data_piece)): 
        par_info, par = data_piece[i]
        encoding = tokenizer.encode_plus(par_info, par, add_special_tokens = True, truncation = True, pad_to_max_length=False, return_attention_mask = True)
        input_ids = encoding["input_ids"]
        par_len_list.append((len(input_ids), i + start_num))
        if i%20000 == 0:
            print("processing len", number, num_proc, i)
    len_dict[num_proc] = par_len_list
    
num_proc = 10
chunk_size = len(data) // 10 + int(len(data) % 10 > 0)

procs = []
manager = mp.Manager()
par_len_dict = manager.dict()
for i in range(10):
    proc = mp.Process(target=run, args=(i, data[i*chunk_size:(i+1)*chunk_size], i*chunk_size, par_len_dict))
    proc.start()
    procs.append(proc)
for proc in procs:
    proc.join()
    
par_len_list = []

for i in range(10):
    for par_len in par_len_dict[i]:
        par_len_list.append(par_len)

par_len_list = sorted(par_len_list, key=lambda x: x[0])

streams = {i: [] for i in range(10)}

for i in range(chunk_size):
    for j in range(10):
        numb = i*10 + j
        if numb < len(data):
            streams[j].append((par_len_list[numb][1], par_len_list[numb][0], data[par_len_list[numb][1]]))


def run_batch(num_proc, par_list, batches_dict):
    batches_list = []
    cur_num = 0
    batches_num = 0
    while True:
        nums_batch = []
        elem_batch = []
        final = False
        if cur_num + 40 < len(par_list) and par_list[cur_num + 40][1] < 100:
            for num, _, elem in par_list[cur_num:cur_num + 40]:
                nums_batch.append(num)
                elem_batch.append(elem)
            cur_num += 40
        elif cur_num + 20 < len(par_list) and par_list[cur_num + 20][1] < 150:
            for num, _, elem in par_list[cur_num:cur_num + 20]:
                nums_batch.append(num)
                elem_batch.append(elem)
            cur_num += 20
        elif cur_num + 10 < len(par_list):
            for num, _, elem in par_list[cur_num:cur_num + 10]:
                nums_batch.append(num)
                elem_batch.append(elem)
            cur_num += 10
        else:
            for num, _, elem in par_list[cur_num:]:
                nums_batch.append(num)
                elem_batch.append(elem)
            final = True
        
        encoding = tokenizer.batch_encode_plus(elem_batch, add_special_tokens = True,
                           pad_to_max_length=True, return_attention_mask = True)
        par_input_ids = encoding["input_ids"]
        par_attention_mask = encoding["attention_mask"]
        par_token_type_ids = encoding["token_type_ids"]
        if len(par_input_ids[0]) > 490:
            par_input_ids = [elem[:490] for elem in par_input_ids]
            par_attention_mask = [elem[:490] for elem in par_attention_mask]
            par_token_type_ids = [elem[:490] for elem in par_token_type_ids]
        new_encoding = {"input_ids": par_input_ids, "attention_mask": par_attention_mask, "token_type_ids": par_token_type_ids}
        batches_list.append([nums_batch, new_encoding])
        if batches_num % 2000 == 0:
            print("run batch", number, num_proc, batches_num)
        batches_num += 1
        
        if final:
            break
    
    print("num_proc", num_proc, "i", i)
    batches_dict[i] = batches_list


procs = []
manager = mp.Manager()
batches_dict = manager.dict()
for i in range(10):
    proc = mp.Process(target=run_batch, args=(i, streams[i], batches_dict))
    proc.start()
    procs.append(proc)
for proc in procs:
    proc.join()

embs = {}

for i in range(10):
    for n, (nums_batch, encoding) in enumerate(batches_dict[i]):
        res = par_emb(encoding)
        for j in range(len(res)):
            embs[nums_batch[j]] = res[j]
        if n%2000 == 0:
            print("final processing", number, i, n)
        
print("processed")

embs_list = []
for i in range(len(data)):
    embs_list.append(embs[i])
        
out = open(f"/archive/evseev/RuWiki/wiki_archives/wikipedia_embeddings_bienc_long/emb_{number}.pickle", 'wb')
pickle.dump(embs_list, out)
out.close()

print("finished")
