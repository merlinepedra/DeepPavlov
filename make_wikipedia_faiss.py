import faiss
import numpy as np
import os
import pickle
import time
from faiss.contrib.ondisk import merge_ondisk
from sklearn.feature_extraction.text import TfidfVectorizer


index_dim = 768

embs = []

for i in range(10):
    fl = open(f"/archive/evseev/RuWiki/wiki_archives/wikipedia_embeddings_bienc_long/emb_{i}.pickle", 'rb')
    data = pickle.load(fl)
    for elem in data:
        embs.append(elem)
    print("read", i)
        
embs = np.array(embs).astype(np.float32)

faiss_index = faiss.index_factory(index_dim, "IVF1000,Flat", faiss.METRIC_INNER_PRODUCT)
faiss_index.train(embs)

print("trained")

faiss_index.add(embs)

print("added")

faiss.write_index(faiss_index, "/archive/evseev/RuWiki/wiki_archives/wikipedia_faiss_long/vectors.index")
        
print("finished")       
