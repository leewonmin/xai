import os
from vllm import LLM, SamplingParams
import pandas as pd
import ast
import torch, gc
import re
import json
from itertools import islice
import time

### 1. vLLM으로 모델 로딩
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_TOKEN"] = "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"
print(os.environ.get("HF_TOKEN"))

model_id = "AlphaGaO/Qwen3-14B-GPTQ"
hf_token = os.environ["HF_TOKEN"]

llm = LLM(
    model=model_id,
    hf_token=hf_token,
    trust_remote_code=True
)

### 2. 데이터 로딩
tmp = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/kisti/data/tmp3.csv') 


### 3. 프롬프트 구성
def clip_text(x, n=200): #paper/patent 자르기
    if not isinstance(x, str):
        return ""
    x = x.strip()
    return x if len(x) <= n else x[:n] + "..."

SYSTEM_PROMPT = (
    "너는 추천 시스템의 '추천 이유'를 설명하는 한국어 AI야. "
    "각 과제가 해당 회사에 왜 적합한지 설명해. "
    "설명은 반드시 한국어로 작성하고 한자 표기는 절대 사용하지마"
    "입력 정보에 있는 내용만 사용해서 설명해. 추측하거나 없는 사실을 만들면 안 돼."
    "너의 생각 과정은 출력하지 말고 최종 답변만 출력해."
    "작성 절차는 다음 순서를 내부적으로 따르되, 절차 자체는 출력하지 마"
    "1. 회사 목적/관심사 요약 → 2. 과제 핵심과 연결 → 3. sim_keyword리스트에서 sim 값 언급 없이 회사 키워드와 과제 키워드의 연관성을 설명해. 리스트가 빈값이면 키워드 관련한 문장은 생성하지마. → 4. project_score 또는 cosine_distance의 실제 값을 포함하고 의미를 해석 → 5. 관련 논문 또는 특허가 있으면 근거로 언급 → 6. 3~4문장으로 설명 생성."
)

def build_company_single_prompt(company_row, rec_row):
    c_id = company_row["company_id"]
    c_name = company_row["company_name"]
    c_score = company_row["company_score"]
    c_purpose = company_row.get("company_purpose", "")
    c_keyword = company_row.get("키워드_company", "")

    pid = rec_row["project_id"]
    pname = rec_row["project_name"]
    pscore = rec_row.get("project_score", "")
    dist = rec_row.get("cosine_distance", "")

    paper = rec_row.get("paper", "")
    patent = rec_row.get("patent", "")
    keyword_proj = rec_row.get("keyword_project", "")

    paper = "" if pd.isna(paper) else clip_text(paper, 200)
    patent = "" if pd.isna(patent) else clip_text(patent, 200)

    sim_keyword_all = rec_row.get("keyword_links", []) or []
    #print('sim_keyword_all: ', sim_keyword_all)
    # 문자열 -> 리스트 파싱
    if isinstance(sim_keyword_all, str):
        try:
            sim_keyword_all = ast.literal_eval(sim_keyword_all)
        except:
            sim_keyword_all = []
    # sim>=0.70 필터
    sim_keyword = []
    for x in sim_keyword_all:
        if not isinstance(x, dict):
            continue
        try:
            if float(x.get("sim", -1)) >= 0.70:
                sim_keyword.append(x)
        except:
            pass
    #print('sim_keyword: ', sim_keyword)


    user_prompt = f"""

    [회사]
    - company_id: {c_id}
    - 회사명: {c_name}
    - company_score: {c_score}
    - company_purpose: {c_purpose}
    - 키워드_company: {c_keyword}

    [추천 과제]
    - 과제번호: {pid}
    - 과제명: {pname}
    - project_score: {pscore}
    - cosine_distance: {dist}
    - 키워드_project: {keyword_proj}
    {f"- 관련 논문: {paper}" if paper else ""}
    {f"- 관련 특허: {patent}" if patent else ""}

    [키워드]
    - sim_keyword: {sim_keyword}

    [출력 규칙]
    - 출력형식:
    [과제번호: <실제 과제번호> | 과제명: <실제 과제명>] - <추천 이유 3~4문장>
    - 출력 형식의 <실제 과제번호>와 <실제 과제명>은 반드시 위에 제공된 [추천 과제]에서 한자 사용하지 말고 문자 그대로 복사해서 작성해
    - 추천 이유에는 반드시 아래 3가지를 모두 포함해:
    1) 회사 목적/관심사와 과제 정보의 연관성을 설명
    2) [키워드]의 sim_keyword 리스트에서 "회사 키워드 '<company>'"와 "과제 키워드 '<project>'" 형식으로 구분해서 그대로 인용하고, 두 키워드의 의미적 유사성과 기술적 관련성을 1문장으로 설명해. sim_keyword 리스트가 빈값이면 키워드 관련한 문장은 아예 출력하지 마.
    3) project_score 또는 cosine_distance의 실제 값을 포함한 근거 설명
    - project_score와 cosine_distance의 실제값은 숫자를 그대로 사용하되, 반드시 소수점 둘째 자리까지만 출력해.
    - project_score은 반드시 평균값인 70점을 기준으로 점수가 높은지 낮은지 그 의미를 포함해서 설명해. 평균값을 기준으로 높을수록 좋은거야
    - cosine_distance는 값이 작을수록 유사하다는 의미로만 해석해서 설명해
    - '관련 논문/관련 특허'는 해당 항목에 값이 있을 때 반드시 근거로 언급하고, 없으면 언급하지 마.
    - 설명, 예시, 참고, 제출 등 어떤 메타 표현도 포함하지 마.
    """.strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


### 4. LLM 모델 호출 & 설명 생성 
def _messages_to_prompt(messages):
    '''
    vLLM 프롬프트 문자열로 변환
    assistant 시작 토큰을 넣어줘야 모델이 답을 출력하기 시작
    '''
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        if role == "system":
            parts.append(f"<|im_start|system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|user\n{content}<|im_end|>")

    # 답변 시작 지점
    parts.append("<|im_start|assistant\n최종 답변만 출력:\n")
    return "\n".join(parts)

@torch.inference_mode()
def generate_explanation(messages, tokenizer, model,
                         max_new_tokens=1024, temperature=0.3, top_p=0.9):
    #tokenizer/model 파라미터는 기존 호출 호환을 위해 남겨둠(사용X).
    
    prompt = _messages_to_prompt(messages)

    params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=1024,
        stop=["<|im_end|>"]
    )

    outputs = llm.generate([prompt], params)
    result = outputs[0].outputs[0].text.strip()
    #print('check:::: ', result[:1000])

    # think 태그 제거
    result = re.sub(r"<think>.*?</think>\s*", "", result, flags=re.DOTALL).strip()
    result = re.sub(r"^\s*</think>\s*", "", result).strip()
    #print('check22:::: ', result[:1000])
    return result


### 5. 설명 생성 실행
def hanja(text):
  return bool(re.search(r'[\u4e00-\u9fff]', text))

out_path = '/content/drive/MyDrive/Colab Notebooks/kisti/result/qwen_GPTQ.csv'
if os.path.exists(out_path):
  os.remove(out_path)
file_exists = os.path.exists(out_path) #False

# sort=False: 회사 그룹을 company_id 오름차순으로 재정렬하지 않게
for company_id, block in tmp.groupby("company_id", sort=False):
    company_time_start = time.time()

    company_name = block["company_name"].iloc[0]
    company_row = block.iloc[0]
    explanations = []
    company_rows = []
    for _, rec_row in block.iterrows():
        messages = build_company_single_prompt(company_row, rec_row)
        one = generate_explanation(messages, tokenizer=None, model=None)

        # 한자 생성될 경우
        retry = 0
        while hanja(one) and retry<2:
          print("한자 감지 : ", one)
          one = generate_explanation(messages, tokenizer=None, model=None,
                                     temperature = max(0.001, 0.3 - 0.1*(retry+1)))
          retry += 1

        explanations.append(one)
        print('one: ', one)

        company_rows.append({
              "company_id": company_id,
              "company_name": block["company_name"].iloc[0],
              "explanation": one
          })

    company_time = time.time()-company_time_start
    print(f"company_name: {company_name} // time: {time.strftime('%H:%M:%S', time.gmtime(company_time))}")
    print(f"\n===== company_id: {company_id} | company_name: {company_name} =====")


    explain_df = pd.DataFrame(company_rows)
    explain_df.to_csv(out_path, mode="a", header=not file_exists, index=False,
        encoding='utf-8-sig')
    file_exists = True