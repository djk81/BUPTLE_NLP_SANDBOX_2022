# -*- coding: utf-8 -*-

import time

import pandas as pd
from torch.nn import CosineSimilarity
from sklearn.metrics.pairwise import euclidean_distances

data = pd.read_csv('similar_sent_and_score_per_johang.csv', low_memory=False, error_bad_lines=False)
data.head(10)

data['원본'].isnull().sum()

len(data)

#!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'

#!pip install transformers==4.8.2

#!pip install sentencepiece

import torch
from kobert_tokenizer import KoBERTTokenizer
from transformers import AutoTokenizer, AutoModel, BertModel

device = torch.device("cuda:0")

data['kobert_코사인유사도'] = 0.0

cos = CosineSimilarity(dim=1, eps=1e-6)

from tqdm import tqdm
from IPython.display import clear_output

@profile
def get_cossim(data, model_name):
  if model_name == 'kobert':
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    model = BertModel.from_pretrained('skt/kobert-base-v1')

  else:
    tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-Medium", do_lower_case=False)
    model = AutoModel.from_pretrained("snunlp/KR-Medium")

  for i in tqdm(range(len(data))):
    # inputs = tokenizer.batch_encode_plus(['본 계약의 효력기간 중에 계약 내용을 변경할 필요가 있는 경우 양 당사자는 상호 서면 합의에 의하여 계약의 내용을 변경할 수 있다'])
    inputs = tokenizer.batch_encode_plus([data['원본'][i]])
    wonbon = model(input_ids = torch.tensor(inputs['input_ids']),
              attention_mask = torch.tensor(inputs['attention_mask']))
    inputs = tokenizer.batch_encode_plus([data['유사문장'][i]])
    temp = torch.tensor(inputs['input_ids'])
    temp2 = torch.tensor(inputs['attention_mask'])
    # yusa = model(input_ids = torch.tensor(inputs['input_ids']),
    #          attention_mask = torch.tensor(inputs['attention_mask']))
    yusa = model(input_ids=temp, attention_mask=temp2)
    cosine = cos(wonbon.pooler_output, yusa.pooler_output)

    if model_name == 'kobert':
      data['kobert_코사인유사도'][i] = cosine.item()

    else:
      data['medium_코사인유사도'][i] = cosine.item()

    clear_output(wait=True)
  return data.sort_values(by=['kobert_코사인유사도'], ascending=False)

start_time = time.time()

data = get_cossim(data, 'kobert')

end_time = time.time()

print("working time : {} sec".format(end_time-start_time))

data.to_csv('johang_similarities.csv')
