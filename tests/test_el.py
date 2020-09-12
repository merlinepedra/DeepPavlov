#pip install git+https://github.com/deepmipt/DeepPavlov.git@entity_linking_test

import json
import re

from deeppavlov import configs, build_model
from deeppavlov.core.data.utils import download
from deeppavlov.core.commands.utils import expand_path

url = "https://raw.githubusercontent.com/AskNowQA/LC-QuAD2.0/master/dataset/test.json"
download_path = expand_path("~/.deeppavlov/downloads/LC-QUAD2.0/test.json")
download(download_path, url)

with open(download_path, 'r') as fl:
    data = json.load(fl)

model = build_model(configs.kbqa.entity_linking_eng_lcquad, download=False)

samples = []

inter = 0
found = 0
total = 0

for n, sample in enumerate(data):
    question = sample["question"]
    replace_tokens = [(' - ', '-'), (' .', ''), ('{', ''), ('}', ''), ('  ', ' '), ('"', "'"), ('(', ''),
                      (')', ''), ('–', '-')]
    for old, new in replace_tokens:
        question = question.replace(old, new)
    if question[-1] != "?":
        question += "?"
    query = sample["sparql_wikidata"]
    entities_list = []
    entities = re.findall(r'wd:(Q\d*)', query)
    res = model([question])
    if res[2][0]:
        inter += len(set(res[2][0]).intersection(set(entities)))
        found += len(res[2][0])
        total += len(entities)
        
precision = inter/found
recall = inter/total
f1 = 2*precision*recall/(precision+recall)
print(f1)

