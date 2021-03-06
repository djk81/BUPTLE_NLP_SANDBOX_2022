"""
SKTBrain 의 KoBERT 를 사용하여 유사도를 탐색하는 가장 기본적인 기능을 구현한다.
:author: 김동진
:date: 2022/07/12
:참조: https://github.com/SKTBrain/KoBERT/tree/master/kobert_hf
"""

# 기본 라이브러리
import json
import time
import pathlib
from datetime import date
from pstats import Stats

import torch
import requests
import line_profiler

from cProfile import Profile
from torch.nn import CosineSimilarity
from tqdm import tqdm
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer

profile = line_profiler.LineProfiler()

# 모델 초기화
model = BertModel.from_pretrained('skt/kobert-base-v1')
# 토크나이저 초기화
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
# cos 초기화
cos = CosineSimilarity(dim=1, eps=1e-6)



@profile
def basic_test_by_list(list_1, list_2):
    """
    리스트로 basic_test 반복 수행
    :param list_1: 문장 리스트1
    :param list_2: 문장 리스트2
    :return:
    """
    result = []
    for one in tqdm(list_1):
        for another in tqdm(list_2):
            result_one = basic_test(one, another)
            result.append({
                "P": result_one,
                "1": one,
                "2": another
            })

    newlist = sorted(result, key=lambda d: d["P"])#[-20:]

    print("RESULT =================================")
    result_file_name = str(int(round(time.time() * 1000))) + '.txt'
    with open( result_file_name, 'w', encoding="UTF-8") as f:
        for log in newlist:
            #print(log['P'])
            #print(log['1'])
            #print(log['2'])

            f.writelines(str(log['P'])+"\n")
            f.writelines(log['1']+"\n")
            f.writelines(log['2']+"\n\n")
    print("[최종종료] ===============================> {}".format(result_file_name))
    return result


@profile
def basic_test(text_1, text_2):
    """
    두 개의 문장으로 유사도 측정
    :param text_1: 문장 1
    :param text_2: 문장 2
    :return:
    """
    #print("START")
    out_1 = get_pooler_output(text_1)
    out_2 = get_pooler_output(text_2)
    cosine = cos(out_1.pooler_output, out_2.pooler_output)
    result = cosine.item()

    #print(text_1)
    #print(text_2)
    print(result)
    #print("END")

    return result


@profile
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
    #print("inputs : {}".format(inputs))

    # attention_mask 도출 -> torch -> attention_mask
    attention_mask = torch.tensor(inputs['attention_mask'])

    # 문장수준 벡터로 도출
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    # print("out : {}".format(out_1))
    # print("out.pooler_output : {}".format(out_1.pooler_output))
    # print("out.pooler_output.shape : {}".format(out.pooler_output.shape))

    return out


def send_request_split_to_sentence(text):
    """
    문장 쪼개기 API
    일일 5천건, 단건 1만글자 제한
    :param text:
    :return:
    """
    params = {
        "access_key": "ac9cb039-e741-4e47-84d3-4c43a26a1e3e",
        "argument": {
            "text": text,
            "analysis_code": "ner"
        }
    }

    response = requests.post("http://aiopen.etri.re.kr:8000/WiseNLU",
                             headers={"Content-type": "application/json"},
                             data=json.dumps(params).encode('utf-8'),
                             verify=False)

    #print(response.text)
    json_rtn = json.loads(response.text)
    print(json_rtn)
    return json_rtn


# 테스트 시작
if __name__ == '__main__':
    profiler = Profile()

    기사_1 = """
    그 모리스 스타일의벽지도 잘 보였고, 점잖은 빛깔의 크레톤 천을 씌운 의자는 사라졌으며, 그전에 애슐리가든의 응접실 벽을 장식했던 안들 사라사는 목격 되었다.
    어머니와 어린애는 나와 스트릭랜드의 차남일 것이다.
    """

    print("파일 load 시작")
    txt_file2 = open(str(pathlib.Path().resolve()) + "/input2.txt", encoding="UTF-8")

    기사_2_리스트 = []
    for line in txt_file2:
        if '@@@' not in line:
            기사_2_리스트.append(line)
    print("파일 load 완료")

    # 문장 쪼개기 1
    json_rtn_1 = send_request_split_to_sentence(기사_1)
    list_1 = []
    for x in tqdm(json_rtn_1['return_object']['sentence']):
        list_1.append(x['text'].strip())

    # 문장 쪼개기 2 - 1만라인 미만
    json_rtn_2 = []
    for out_1depth in tqdm(기사_2_리스트):
        json_rtn_2.append(send_request_split_to_sentence(out_1depth))

    for tmp in tqdm(json_rtn_2):
        list_2 = []
        for x in tqdm(tmp['return_object']['sentence']):
            list_2.append(x['text'].strip())

    # 실행 & 성능 측정
    basic_test_by_list(list_1, list_2)


    # profiler.runcall(
    #     basic_test_by_list,
    #     list_1,
    #     list_2
    # )

    # 성능 로그
    # stats = Stats(profiler)
    # stats.strip_dirs()
    # stats.sort_stats('cumulative')
    # stats.print_stats()
