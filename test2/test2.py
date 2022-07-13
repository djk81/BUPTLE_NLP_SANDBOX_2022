"""
SKTBrain 의 KoBERT 를 사용하여 유사도를 탐색하는 가장 기본적인 기능을 구현한다.
:author: 김동진
:date: 2022/07/12
:참조: https://github.com/SKTBrain/KoBERT/tree/master/kobert_hf
"""

# 기본 라이브러리
from pstats import Stats

import torch
from cProfile import Profile
from torch.nn import CosineSimilarity
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer


# 모델 초기화
model = BertModel.from_pretrained('skt/kobert-base-v1')
# 토크나이저 초기화
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
# cos 초기화
cos = CosineSimilarity(dim=1, eps=1e-6)


def basic_test(text_1, text_2):
    """
    두 개의 문장으로 유사도 측정
    :param text_1: 문장 1
    :param text_2: 문장 2
    :return:
    """
    print("START")
    out_1 = get_pooler_output(text_1)
    out_2 = get_pooler_output(text_2)
    cosine = cos(out_1.pooler_output, out_2.pooler_output)
    result = cosine.item()

    print(text_1)
    print(text_2)
    print(result)
    print("END")


def get_pooler_output(text):
    """
    KoBERT 모델사용하여 cosine 유사도를 측정할 수 있는. pooler_output 을 리턴한다.
    :param text:
    :return:
    """
    # inputs 도출
    inputs = tokenizer.batch_encode_plus([text])

    # input_ids 도출 -> torch -> input_ids
    input_ids = torch.tensor(inputs['input_ids'])
    print("inputs : {}".format(inputs))

    # attention_mask 도출 -> torch -> attention_mask
    attention_mask = torch.tensor(inputs['attention_mask'])

    # 문장수준 벡터로 도출
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    # print("out : {}".format(out_1))
    # print("out.pooler_output : {}".format(out_1.pooler_output))
    # print("out.pooler_output.shape : {}".format(out.pooler_output.shape))

    return out


# 테스트 시작
if __name__ == '__main__':
    profiler = Profile()
    
    # 실행 & 성능 측정
    profiler.runcall(
        basic_test,
        "가나다라마바사",
        "렘입숨은 전통 가나다라마바사 라틴어와 닮은 점 때문에 종종 호기심을 유발하기도 하지만 그 이상의 의미를 담지는 않는다.문서에서 텍스트가 보이면 사람"
     )

    # 성능 로그
    stats = Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()
