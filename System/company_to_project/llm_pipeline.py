import gc
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from data_pipeline import clean_text, parse_struct, to_list

'''프롬프트 구성'''

import ast
import json
import numpy as np
import pandas as pd

def to_py(x):
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_py(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def normalize_conduct_list_company(x):
    x = parse_struct(x)

    if not isinstance(x, list):
        return []

    result = []

    for item in x:
        if not isinstance(item, dict):
            continue

        info = parse_struct(item.get("company_info"))
        if not isinstance(info, dict):
            info = {}

        company_id = clean_text(item.get("company_id"))
        fallback_name = clean_text(item.get("company_name"))

        company_info = {
            "company_name": clean_text(fallback_name),
            "region": clean_text(info.get("region", None)),
            "company_keyword": normalize_str_list(info.get("company_keyword", [])),
            "company_purpose_list": normalize_str_list(info.get("company_purpose_list", [])),
            "company_patent": normalize_str_list(info.get("company_patent", [])),
        }

        has_info = any([
            company_info["company_name"],
            company_info["region"],
            company_info["company_keyword"],
            company_info["company_purpose_list"],
            company_info["company_patent"],
        ])

        if company_id or has_info:
            result.append({
                "company_id": company_id,
                "company_name": company_info["company_name"],
                "company_info": company_info
            })

    return result


def normalize_conduct_list_project(x):
    x = parse_struct(x)

    if not isinstance(x, list):
        return []

    result = []

    for item in x:
        if not isinstance(item, dict):
            continue

        info = parse_struct(item.get("project_info"))
        if not isinstance(info, dict):
            info = {}

        project_id = clean_text(item.get("project_id"))
        fallback_name = clean_text(item.get("project_name") or item.get("과제명"))

        project_info = {
            "project_name": clean_text(info.get("project_name") or info.get("과제명") or fallback_name),
            "project_keyword": normalize_str_list(info.get("project_keyword") or info.get("키워드_project") or []),
            "paper": normalize_str_list(info.get("paper", None)),
            "patent": normalize_str_list(info.get("patent", None)),
        }

        has_info = any([
            project_info["project_name"],
            project_info["project_keyword"],
            project_info["paper"],
            project_info["patent"],
        ])

        if project_id or has_info:
            result.append({
                "project_id": project_id,
                "project_name": project_info["project_name"],
                "project_info": project_info
            })

    return result


def normalize_str_list(x):
    return to_list(x, keep_scalar=True, string_only=True)

def extract_group(x): 
    x = parse_struct(x)

    if not isinstance(x, dict):
        return None

    group = x.get("group")

    if group is None:
        return None

    s = str(group).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None

    return s

REGION_GROUPS = {
    "수도권": ["서울", "경기", "인천"],
    "충청권": ["충남", "충북", "대전", "세종"],
    "호남권": ["전북", "전남", "광주"],
    "영남권": ["경남", "경북", "대구", "부산"],
    "강원권": ["강원"],
}

REGION_TO_GROUP = {}
for group_name, regions in REGION_GROUPS.items():
    for r in regions:
        REGION_TO_GROUP[r] = group_name


def normalize_region(x):
    if x is None:
        return ""
    return str(x).strip()


def analyze_region_relation(company_region, conduct_list_company):
    company_region = normalize_region(company_region)
    company_group = REGION_TO_GROUP.get(company_region, "")

    result = {
        "company_region": company_region,
        "company_region_group": company_group,
        "same_region_count": 0,
        "similar_region_count": 0,
        "same_region_companies": [],
        "similar_region_companies": [],
        "same_region_names": [],
        "similar_region_names": [],
        "same_region_group": "",
        "similar_region_group": "",
        "has_same_region": False,
        "has_similar_region": False,
    }

    if not company_region:
        return result

    same_region_companies = []
    similar_region_companies = []

    for item in conduct_list_company:
        if not isinstance(item, dict):
            continue

        info = item.get("company_info", {})
        if not isinstance(info, dict):
            info = {}

        c_name = clean_text(item.get("company_name") or info.get("company_name"))
        c_region = normalize_region(info.get("region"))

        if not c_region:
            continue

        company_item = {
            "company_name": c_name,
            "region": c_region
        }

        # 1) 정확히 같은 지역
        if c_region == company_region:
            same_region_companies.append(company_item)
            continue

        # 2) 강원은 exact only
        if company_region == "강원" or c_region == "강원":
            continue

        # 3) 같은 권역이면 유사 지역
        c_group = REGION_TO_GROUP.get(c_region, "")
        if company_group and c_group and company_group == c_group:
            similar_region_companies.append(company_item)

    result["same_region_count"] = len(same_region_companies)
    result["similar_region_count"] = len(similar_region_companies)
    result["same_region_companies"] = same_region_companies
    result["similar_region_companies"] = similar_region_companies
    result["same_region_names"] = [x["company_name"] for x in same_region_companies if x["company_name"]]
    result["similar_region_names"] = [x["company_name"] for x in similar_region_companies if x["company_name"]]
    result["same_region_group"] = company_group if same_region_companies else ""
    result["similar_region_group"] = company_group if similar_region_companies else ""
    result["has_same_region"] = len(same_region_companies) > 0
    result["has_similar_region"] = len(similar_region_companies) > 0

    return result

FORBIDDEN_PATTERNS = [
    r"제공되지",
    r"확인되지",
    r"존재하지",
    r"존재하지\s*않",
    r"없어",
    r"없으",
    r"없다",
    r"없는",
    r"비어\s*있",
    r"정보가\s*없",
    r"데이터가\s*없",
    r"이력.*없",
    r"수행.*없",
    r"부재",
    r"작성하지\s*않",
    r"작성할\s*수\s*없",
    r"언급할\s*수\s*없",
    r"반영할\s*수\s*없",
    r"분석.*제한적",
]

def normalize_section_text(v):
    if isinstance(v, list):
        return " ".join(str(x).strip() for x in v if str(x).strip())
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v).strip()


def postprocess_output(obj, expected_keys):
    cleaned = {}

    for k in expected_keys:
        if k not in obj:
            continue

        text = normalize_section_text(obj[k])

        sentences = re.split(r'(?<=[.!?。])\s*', text)

        kept = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if any(re.search(p, s) for p in FORBIDDEN_PATTERNS):
                continue
            kept.append(s)

        cleaned_text = " ".join(kept).strip()

        if cleaned_text:
            cleaned[k] = cleaned_text

    final_text = json.dumps(cleaned, ensure_ascii=False)

    is_valid = (
        set(cleaned.keys()) == set(expected_keys)
        and not any(re.search(p, final_text) for p in FORBIDDEN_PATTERNS)
    )

    return cleaned, is_valid

def clean_forbidden_text(text):
    sentences = re.split(r'(?<=[.!?。])\s*', str(text).strip())

    kept = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if any(re.search(p, s) for p in FORBIDDEN_PATTERNS):
            continue
        kept.append(s)

    return " ".join(kept).strip()

def count_prompt_tokens(messages, tok):
    prompt = _messages_to_prompt(messages, tok)
    ids = tok(prompt, add_special_tokens=False).input_ids
    return len(ids)


SYSTEM_PROMPT = (
    "너는 추천 시스템의 '추천 이유'를 설명하는 한국어 AI야. "
    "입력 JSON의 정보를 기반으로 특정 회사에 특정 과제가 왜 추천되었는지 한국어로 설명한다. "
    "기술적 원리나 설명이 필요한 경우에는 일반적인 산업/기술 지식을 활용해 쉽게 풀어 설명할 수 있다. "
    "다만 입력 정보와 무관한 새로운 사실(특정 기업의 추가 사업, 특정 수치, 특정 기술 보유 등)을 만들어서는 안 된다."
    "아래 규칙을 기반으로 최종 결과만 생성해. 중간 추론 과정은 출력하지 마.\n"
    "1) 회사 정보 해석\n"
    "   1-1) company.purpose는 유효한 값이 있는 경우에만 해석하되, company.keyword와 기술적으로 연결되는 항목만 선택적으로 사용하며, 연결성이 낮은 항목은 설명에서 제외한다.\n"
    "   1-2) company.keyword는 유효한 값이 있는 경우에만 의미 단위로 묶어 회사의 핵심 역량과 기술 요소를 도출한다.\n"
    "   1-3) company.company_patent는 유효한 값이 있는 경우에만 검토하여 회사의 기술 축, 구현 방식, 적용 가능 분야를 파악한다.\n"
    "2) 과제 정보 해석\n"
    "   2-1) project.title, project.keywords는 유효한 값이 있는 경우에만 과제의 핵심 대상과 기술 방향을 추출한다.\n"
    "   2-2) related_research.paper 또는 related_research.patent에 유효한 값이 있는 경우에만 집중하는 핵심 기술 또는 연구 방향을 파악한다.\n"
    "3) 1)과 2)의 요약 내용을 근거로 회사와 과제의 연관성을 '연관성' 섹션에서 설명한다.\n"
    "   - 회사의 사업 및 기술 역량을 바탕으로 과제의 핵심 목적과 기술이 회사의 기술, 제품, 제조 공정 또는 연구개발과 어떻게 연결되는지 설명한다.\n"
    "   - 연결성 설명에는 반드시 1문장 이상으로 과제의 핵심 기술/방법이 왜 필요한지(작동 원리, 메커니즘, 해결하려는 문제의 원인)를 일반적인 기술 지식으로 풀어서 설명한다.\n"
    "   - 기술 설명은 '조건 또는 변수 → 그 변화로 발생하는 문제 또는 결과 → 그래서 해당 기술이 필요함'의 인과 구조로 설명한다.\n"
    "   - 연결성 설명은 다음 순서를 따른다: (회사 기술이 과제 내용에서 수행하는 역할과 의미) → (해당 기술이 실제로 활용되거나 적용될 수 있는 중간 단계) → (회사 기술/사업과의 연결) \n"
    "   - 회사명은 반드시 입력 JSON의 company.name 값을 그대로 사용한다.\n"
    "4) 기술 설명이 필요한 경우에는 일반적인 산업 공정 지식이나 기술 지식을 활용하여 설명할 수 있다.\n"
    "   다만 입력 JSON에 없는 회사 사실이나 과제 정보를 새로 만들어서는 안 된다.\n"
    "5) 재무 및 기술 유망성, 기술 성숙도를 바탕으로 '추천 과제의 우수성' 섹션에서 작성한다.\n"
    "    - project.총연구비_상위비율과 project.총연구비_group 값이 모두 있을 때만 언급한다.\n"
    "    - project.총연구비_group이 '전체'이면 '총연구비가 전체의 상위 n% 수준'으로, 그 외에는 '총연구비가 {group} 업종 내 상위 n% 수준'으로 표현한다.\n"
    "    - has_paper가 true이면 paper_list_count의 수만큼 논문 실적이 있음을 설명한다.\n"
    "    - has_patent가 true이면 patent_list_count의 수만큼 특허 실적이 있음을 설명한다.\n"
    "    - has_paper가 false이면 논문 실적 관련 내용을 어떠한 형태로도 작성하지 않는다.\n"
    "    - has_patent가 false이면 특허 실적 관련 내용을 어떠한 형태로도 작성하지 않는다.\n"
    "    - has_paper와 has_patent가 모두 false이면 논문·특허 실적 및 이를 근거로 한 기술 연관성 관련 내용을 어떠한 형태로도 작성하지 않는다.\n"
    "    - company.벤처기업여부, company.이노비즈여부, company.메인비즈여부, company.ASTI 여부, company.특구 여부 중 값이 'Y'인 항목만 선택하여 회사의 인증/지정 근거로 반영한다.\n"
    "    - 언급 시 반드시 다음 명칭을 그대로 사용한다: company.ASTI 여부=ASTI 회원사, company.특구 여부=특구 지정기업, company.벤처기업여부=벤처기업, company.이노비즈여부=이노비즈, company.메인비즈여부=메인비즈.\n"
    "    - 위 명칭은 다른 표현으로 변경하거나 축약하지 않는다.\n"
    "    - company.매출성장율_상위비율과 company.매출성장율_group 값이 모두 있을 때만 언급한다.\n"
    "    - company.매출성장율_group이 '전체'이면 '매출성장율이 전체의 상위 n% 수준'으로, 그 외에는 '매출성장율이 {group} 업종 내 상위 n% 수준'으로 표현한다.\n"
    "    - company.영업이익율_상위비율과 company.영업이익율_group 값이 모두 있을 때만 언급한다.\n"
    "    - company.영업이익율_group이 '전체'이면 '영업이익율이 전체의 상위 n% 수준'으로, 그 외에는 '영업이익율이 {group} 업종 내 상위 n% 수준'으로 표현한다.\n"
    "    - company.연구개발비_상위비율과 company.연구개발비_group 값이 모두 있을 때만 언급한다.\n"
    "    - company.연구개발비_group이 '전체'이면 '연구개발비가 전체의 상위 n% 수준'으로, 그 외에는 '연구개발비가 {group} 업종 내 상위 n% 수준'으로 표현한다.\n"
    "    - company.부채비율_하위비율과 company.부채비율_group 값이 모두 있을 때만 언급한다.\n"
    "    - company.부채비율_group이 '전체'이면 '부채비율이 전체의 상위 n% 수준'으로, 그 외에는 '부채비율이 {group} 업종 내 상위 n% 수준'으로 표현한다.\n"
    "    - 위 비율 정보와 group은 해당 값이 모두 있을 때만 언급하고, 값이 없으면 완전히 생략한다.\n"
    "    - 정량 지표는 숫자를 나열하듯 쓰지 말고, 추천 가능성을 설명하는 보조 근거로 자연스럽게 종합 서술한다.\n"
    "    - 추천 과제 우수성 섹션은 다음 순서로 작성한다.\n"
    "      1. 과제의 총연구비 수준을 기반으로 과제 자체의 우수성을 설명한다.\n"
    "      2. has_paper 또는 has_patent 중 하나 이상이 true인 경우에만 다음 내용을 기반으로 과제의 기술 성숙도를 작성한다.\n"
    "        - has_paper가 true이면 paper_list_count의 수만큼 논문 실적이 있음을 설명한다.\n"
    "        - has_patent가 true이면 patent_list_count의 수만큼 특허 실적이 있음을 설명한다.\n"
    "        - has_paper와 has_patent가 모두 false인 경우에는 논문·특허 실적 및 기술 연관성 관련 내용을 어떠한 형태로도 작성하지 않는다.\n"
    "      3. company.* 인증/지정 항목 중 값이 'Y'인 항목이 있는 경우 회사의 기술·사업 역량 근거로 반영한다.\n"
    "      4. company.* 재무지표가 존재하는 경우 추천 가능성을 뒷받침하는 보조 근거로 자연스럽게 종합 서술한다.\n"
    "      5. 마지막으로 과제 적합성 및 추천 타당성을 종합 결론으로 작성한다.\n"
    "6) 다음 내용을 바탕으로 '유사 사례' 섹션을 생성한다.\n"
    "  [매칭 과제와의 유사성]\n"
    "   - company.has_patent_matching_related가 true인 경우에만, patent_matching_related에 포함된 특허 제목들과 2)에서 파악한 현재 추천 과제의 내용을 비교하여 기술적 연관성을 설명한다.\n"
    "   - 이때 특허 제목을 단순 나열하지 말고, 특허 제목들이 공통적으로 가리키는 기술 주제, 적용 분야, 해결하려는 문제와 현재 추천 과제의 목표·핵심 기술·적용 방식이 어떻게 연결되는지 종합하여 설명한다.\n"
    "   - company.has_patent_matching_related가 false이면 특허 매칭 관련 내용을 어떠한 형태로도 작성하지 않는다.\n"

    " [매칭 과제의 유사 논문 및 특허 성과]\n"
    "    - has_paper 또는 has_patent 중 하나 이상이 true인 경우에만 매칭 과제의 유사 논문 및 특허 성과 내용을 작성한다.\n"
    "    - has_paper와 has_patent가 모두 false이면 매칭 과제의 유사 논문 및 특허 성과 내용을 어떠한 형태로도 작성하지 않는다.\n"
    "    - has_paper가 false이면 논문 실적, 논문 부재, 논문 기반 기술 연관성 관련 내용을 어떠한 형태로도 작성하지 않는다.\n"
    "    - has_patent가 false이면 특허 실적, 특허 부재, 특허 기반 기술 연관성 관련 내용을 어떠한 형태로도 작성하지 않는다.\n"
    "    - 논문 또는 특허가 여러 개 제공되는 경우 각 항목을 단순 나열하지 말고 반복되는 기술 주제 또는 공통 연구 방향을 중심으로 종합적으로 설명한다.\n"
    "    - has_paper 또는 has_patent 중 true인 항목의 내용만 종합하여 해당 과제가 어떤 기술 분야나 연구 방향에 집중하고 있는지 정리하고 회사 목적 및 기술과 어떻게 연관 되는지 설명한다.\n"
    "    - 이후 해당 기술이 일반적으로 어떤 장치나 시스템에서 사용되는지 설명하고, 그 장치 또는 시스템이 1)에서 파악한 현재 회사의 사업과 어떻게 연결되는지 설명한다.\n"
    "    - has_paper 또는 has_patent 중 하나 이상이 true인 경우, 과제의 기술 → 기술이 필요한 이유 → 기술이 사용되는 장치 또는 시스템 → 회사 사업과의 연관 근거 순서로 설명한다.\n"

    "   - conduct_list_company와 conduct_list_project는 각각 독립적으로 판단한다.\n"
    "   - 빈 리스트, None, 누락된 값에 대해서는 어떤 형태로도 언급하지 않는다.\n"
    "   - 빈 값을 근거로 한 추측, 보완 설명, 일반화된 설명을 생성하지 않으며, '없다', '제공되지 않았다', '비어 있다', '확인되지 않는다'와 같은 표현은 절대 생성하지 않는다.\n"

    "   [추천된 과제를 수행한 기업 유사 사례]\n"
    "   - has_conduct_company가 True인 경우에만 추천된 과제를 수행한 기업 유사 사례 내용을 작성한다.\n"
    "   - has_conduct_company가 False이면 수행 기업, 수행 기업 부재, 수행 기업 유사 사례 관련 내용을 어떠한 형태로도 작성하지 않는다.\n"
    "   - conduct_list_company에서는 각 항목의 company_info 안에 있는 company_name, company_purpose_list, company_keyword, region, company_patent 중 값이 존재하는 정보만 활용한다.\n"
    "   - 추천된 과제를 실제로 수행했던 회사들의 company_info와 1)에서 파악한 현재 회사 내용, 2)에서 파악한 추천된 과제 내용을 비교하여 세 요소가 공통적으로 어떤 기술, 사업, 공정, 제품, 연구개발 방향에서 유사한지 설명한다.\n"
    "   - 단순히 '유사하다'고 표현하지 말고, 수행 회사들의 기술 또는 사업 내용이 현재 회사 및 추천 과제와 어떤 점에서 공통적으로 닮아 있는지 구체적으로 서술한다.\n"
    "   - conduct_list_company에 여러 항목이 있을 경우 각 회사를 하나씩 단순 나열하지 말고, 반복되는 공통 기술 주제, 공통 사업 방향, 공통 문제 해결 방식 중심으로 종합하여 설명한다.\n"

    "   [추천 과제를 수행한 기업의 지역적 연관성]\n"
    "   - 지역 근거는 has_conduct_company가 True인 경우에만 보조 근거로 활용할 수 있다.\n"
    "   - has_conduct_company가 False이면 지역적 연관성 관련 내용을 어떠한 형태로도 작성하지 않는다.\n"
    "   - 지역 근거는 기술적 유사성의 대체가 아니라 보조 근거로만 사용하며, 같은 지역 또는 유사 지역이라는 이유만으로 기술적 적합성을 단정하지 않는다.\n"
    "   - conduct_list_company의 각 항목에 포함된 company_info.region과 company.region을 비교하여 지역적 연관성을 판단한다.\n"
    "   - 현재 회사와 동일한 지역의 수행 기업이 있으면 유사 지역보다 우선적으로 설명한다.\n"
    "   - 동일 지역 기업이 여러 개이면 company.region 값을 직접 언급하여 지역적 연관성이 비교적 강한 보조 근거임을 설명한다.\n"
    "   - 동일 지역 기업이 없고 유사 지역 기업이 있을 때만 유사 지역 근거를 사용한다.\n"
    "   - 유사 지역은 수도권(서울, 경기, 인천), 충청권(충남, 충북, 대전, 세종), 호남권(전북, 전남, 광주), 영남권(경남, 경북, 대구, 부산), 강원권(강원) 기준으로 판단한다.\n"
    "   - 강원권은 정확히 '강원'이 일치할 때만 지역 근거로 사용한다.\n"
    "   - region_relation.has_same_region이 true이면 동일 지역 근거를 보조적으로 사용하고, company.region 값을 직접 언급한다.\n"
    "   - region_relation.has_same_region이 false이고 region_relation.has_similar_region이 true이면 유사 권역 근거를 보조적으로 사용하고, region_relation.company_region_group 값을 직접 언급한다.\n"

    "   [기업이 수행한 과제 유사 사례]\n"
    "   - has_conduct_project가 True인 경우에만 기업이 수행한 과제 유사 사례 내용을 작성한다.\n"
    "   - has_conduct_project가 False이면 수행 과제, 수행 이력, 과거 과제 부재와 관련된 내용을 어떠한 형태로도 작성하지 않는다.\n"
    "   - conduct_list_project에서는 각 항목의 project_info 안에 있는 project_name, project_keyword, paper, patent 중 값이 존재하는 정보만 활용한다.\n"
    "   - conduct_list_project에 여러 항목이 있을 경우 각 과제를 하나씩 단순 나열하지 말고, 공통 기술 주제, 공통 연구개발 방향, 공통 문제 해결 방식 중심으로 종합하여 설명한다.\n"
    "   - 현재 회사가 과거에 수행했던 과제들의 project_info와 2)에서 파악한 현재 추천된 과제의 내용을 비교하여 목표, 핵심 기술, 적용 방식, 해결하려는 문제 측면에서 어떤 유사성이 있는지 설명한다.\n"

    "7) 모든 요소를 섹션 구조로 일관되게 정리한다.\n"
    "8) 모든 섹션은 일반인이 이해할 수 있는 수준으로 작성한다.\n\n"
    "[출력 규칙]\n"
    "- 위 내부 사고 단계나 중간 판단, 계산 과정은 절대 출력하지 않는다.\n"
    "- 추측하거나 없는 사실을 만들지 않는다.\n"
    "- 입력 JSON에 값이 없는 항목은 언급하지 않는다.\n"
    "- 다만 기술 설명이나 원리 설명이 필요한 경우에는 일반적인 산업 또는 기술 지식을 활용하여 설명할 수 있다.\n"
    "- 모든 설명은 '평가'가 아니라 '추천 이유 설명'의 관점에서 작성한다.\n"
    )


FEWSHOT_MESSAGES = [
    {"role": "user", "content": (
        "아래 JSON만을 근거로 추천 근거를 작성해.\n"
        "출력은 JSON 하나만 반환해. (JSON 외 텍스트 금지)\n"
        "키는 output_requirements.format에 있는 섹션 제목을 그대로 사용해.\n\n"
        + json.dumps({
            "company": {
                "company_id": "C001",
                "name": "샘플회사_알파12a",
                "purpose": ['반도체 및 평판디스플레이 제조용 기계 제조업', '항공기, 우주선 및 보조장치 제조업', '공학 연구개발업'],
                "keyword": [
                        '코팅공정용', '나노물질자가정렬방법', '코팅물질', '용액공정', '용액기반', '공압모듈', '코팅장비마스크패터닝', '코팅장비', '코팅솔루션', '필름제조용', '금속입자소결체', '진공증착', '공압부품', '잉크토출장치', '공정기술', '피인쇄물질', '밸브제작', '용액'
                    ],
                "patent_matching_related": [
                        {"특허명":"프린팅 장치"},
                        {"특허명":"유도보조 전극을 포함하는 유도 전기수력학 젯 프린팅 장치"},
                        {"특허명":"피드백 제어형 인쇄 시스템"},
                        {"특허명":"전기수력학 방식의 분사 노즐"}
                ],
                "has_patent_matching_related": True,

                "conduct_list_project": [
                    {
                        "과제명": "고분자 기반 기능성 필름 제조 공정 개발",
                        "과제코드": "P9001",
                        "project_info": {
                            "project_name": "고분자 기반 기능성 필름 제조 공정 개발",
                            "project_keyword": ["고분자", "필름", "코팅", "건조", "표면 제어"],
                            "paper": [],
                            "patent": []
                        }
                    },
                    {
                        "과제명": "정밀 프린팅 기반 소재 패터닝 기술 개발",
                        "과제코드": "P9002",
                        "project_info": {
                            "project_name": "정밀 프린팅 기반 소재 패터닝 기술 개발",
                            "project_keyword": [],
                            "paper": None,
                            "patent": None
                        }
                    }
                ],
                "has_conduct_project": True,

                "region": "서울",

                "벤처기업여부": "Y",
                "이노비즈여부": "Y",
                "메인비즈여부": "N",
                "ASTI 여부": "Y",
                "특구 여부": "Y",
                "매출성장율_상위비율": '5',
                "매출성장율_group": "전체",

                "영업이익율_상위비율": "",
                "영업이익율_group": None,

                "부채비율_하위비율": '10',
                "부채비율_group": "전체",

                "연구개발비_상위비율": '20',
                "연구개발비_group": "전문직별 공사업",

            },
            "project": {
                "project_id": "P001",
                "title": "액정 엘라스토머 기반 4D 프린팅 소재 개발",
                "keyword": [
                        "코팅공정용", "용액공정", "금속입자소결체", "잉크토출장치", "공정기술"
                    ],
                "총연구비_상위비율": '5',
                "총연구비_group": "전체"
            },

            "conduct_list_company": [
                {
                    "company_name": "가상회사_유동제어_1",
                    "company_id": "C101",
                    "company_info": {
                        "company_name": "가상회사_유동제어_1",
                        "region": "서울",
                        "company_keyword": ["기능성 필름", "정밀 코팅", "박막 형성", "공정 제어"],
                        "company_purpose_list": ["디스플레이 및 전자재료용 소재 개발"],
                        "company_patent": []
                    }
                },
                {
                    "company_name": "임시조직a12k",
                    "company_id": "C102",
                    "company_info": {
                        "company_name": "임시조직a12k",
                        "region": "경기",
                        "company_keyword": [],
                        "company_purpose_list": [],
                        "company_patent": []
                    }
                }
            ],
            "has_conduct_company": True,

            "region_relation": {
              "company_region": "서울",
              "company_region_group": "수도권",
              "same_region_count": 1,
              "similar_region_count": 1,
              "same_region_companies": [
                  {"company_name": "가상회사_유동제어_1", "region": "서울"}
              ],
              "similar_region_companies": [
                  {"company_name": "임시조직a12k", "region": "경기"}
              ],
              "same_region_names": ["가상회사_유동제어_1"],
              "similar_region_names": ["임시조직a12k"],
              "same_region_group": "수도권",
              "similar_region_group": "수도권",
              "has_same_region": True,
              "has_similar_region": True
          },
            "related_research": {
                "paper": ["습윤 고분자 탄성 액추에이터의 4D 프린팅"],
                "patent": ["폴리로탁산 가교체를 도입한 액정 엘라스토머 필름의 제조 방법"]
            },
            "paper_list_count" : 1,
            "patent_list_count" : 1,
            "has_paper" : True,
            "has_patent" : True,
            "output_requirements": {
                "language": "ko",
                "format": [
                    "연관성",
                    "추천 과제의 우수성",
                    "유사 사례"
                ],
                "section_sentence_range": "각 섹션 4~6문장",
                "forbidden": ["예시", "참고", "제출", "메타"]
            }
        }, ensure_ascii=False)
    )},

    {"role": "assistant", "content": json.dumps({
        "연관성": (
            "샘플회사_알파12a는 코팅공정, 용액기반 공정, 잉크 토출 장치, 공압 모듈 및 정밀 코팅 장비와 관련된 기술 요소를 보유하고 있어 액정 엘라스토머 기반 4D 프린팅 소재 개발 과제와 높은 기술적 연결성을 가진다."
            "액정 엘라스토머 기반 4D 프린팅은 소재 내부 배열과 코팅 균일도가 변하면 형태 변화 성능과 구동 안정성이 달라질 수 있기 때문에, 용액 공정과 정밀 코팅 기술을 통해 소재의 두께와 배열을 안정적으로 제어하는 기술이 중요하다."
            "샘플회사_알파12a의 코팅장비, 공정기술, 잉크토출장치 관련 역량은 기능성 필름 제조와 정밀 소재 패터닝 과정에서 활용될 수 있으며, 이는 액정 엘라스토머 필름 제조 및 4D 프린팅 공정의 중간 제조 단계와 직접 연결된다."
            "특히 금속입자소결체와 용액기반 공정 기술은 기능성 소재를 균일하게 형성하고 패턴 정밀도를 높이는 데 활용될 수 있어 고분자 기반 액추에이터 제작 공정과의 연관성이 높다."
            "또한 프린팅 장치, 피드백 제어형 인쇄 시스템, 전기수력학 방식의 분사 노즐과 같은 특허들은 정밀 분사 및 인쇄 공정 안정화와 관련된 기술 축을 형성하고 있으며, 이는 4D 프린팅 소재 제조 과정에서 필요한 정밀 패터닝 및 균일 코팅 기술과 자연스럽게 연결된다."
        ),
        "추천 과제의 우수성": (
            "추천된 과제는 총연구비가 전체의 상위 5% 수준에 해당하여 연구개발 규모 측면에서 우수한 과제로 볼 수 있습니다."
            "또한 관련 논문 1건과 특허 1건의 실적이 있어, 형태 변화가 가능한 고분자 소재와 필름 제조 기술에 대한 연구 기반이 확인됩니다."
            "샘플회사_알파12a는 ASTI 회원사이면서 특구 지정기업과 벤처기업, 이노비즈 인증을 보유하고 있어 기술 기반 사업화 역량을 뒷받침할 수 있습니다."
            "재무적으로도 매출성장율이 전체의 상위 5% 수준이고, 부채비율이 전체의 상위 10% 수준이며, 연구개발비가 전문직별 공사업 업종 내 상위 20% 수준으로 나타나 성장성과 연구개발 투입 측면에서 추천 가능성을 보강합니다."
            "종합하면 이 과제는 회사의 고분자·코팅·정밀 제조 역량과 연결될 수 있는 기술 주제를 가지고 있으며, 연구 규모와 관련 성과, 회사의 기술사업화 기반을 함께 고려할 때 추천 타당성이 높습니다.",
        ),
        "유사 사례": (
            "샘플회사_알파12a가 보유한 프린팅 장치, 유도 전기수력학 젯 프린팅 장치, 피드백 제어형 인쇄 시스템, 전기수력학 방식의 분사 노즐 관련 특허는 정밀한 소재 토출과 패턴 형성 기술을 중심으로 하며, 이는 미세한 고분자 소재를 균일하게 제어·분사해 원하는 형태와 구조를 구현해야 하는 현재 추천 과제의 4D 프린팅 소재 제조 기술과 연관됩니다."
            "추천된 과제의 관련 논문과 특허는 습윤 고분자 탄성 액추에이터와 액정 엘라스토머 필름 제조 기술을 중심으로 하고 있어, 형태 변화형 고분자 소재와 기능성 필름 제작 분야에 대한 연구개발이 이루어지고 있음을 보여줍니다."
            "이러한 기술은 자극이나 시간 변화에 따라 형태가 변하는 액추에이터, 기능성 필름, 정밀 패터닝 소재 등에 활용될 수 있으며, 샘플회사_알파12a의 반도체·디스플레이 제조 장비 및 고기능성 소재 제조 사업과 연결될 수 있습니다."
            "추천된 과제를 수행한 기업 중 기능성 필름, 정밀 코팅, 박막 형성, 공정 제어와 같은 키워드가 현재 회사의 코팅·증착·소재 제조 역량 및 추천 과제의 고분자 기반 4D 프린팅 소재 개발 방향과 공통점을 가집니다."
            "또한 추천된 과제를 수행한 기업 중 현재 기업과 동일한 서울 지역 기업이 포함되어 있는 점은 유사 산업 및 기술 환경 내 수행 사례가 존재함을 보여주며, 지역 기반 협업과 기술 연계 가능성 측면에서 추천된 과제가 현재 기업과도 연관성이 있음을 뒷받침하는 근거로 볼 수 있습니다."
            "기업이 수행한 과제 중에서도 고분자 기반 기능성 필름 제조와 정밀 프린팅 기반 소재 패터닝 기술이 반복적으로 나타나, 현재 추천 과제와 목표 소재, 핵심 공정, 적용 방식 측면에서 유사한 연구개발 흐름을 보입니다."
        ),
    }, ensure_ascii=False)}
]



def build_company_single_prompt(company_row, rec_row):
    c_id = company_row["company_id"]
    c_name = company_row["company_name"]
    #c_desc = company_row.get("company_description", "")
    c_purpose = normalize_str_list(company_row.get("company_purpose_list", []))
    c_keyword = normalize_str_list(company_row.get("키워드_company", []))

    pid = rec_row["project_id"]
    pname = rec_row["project_name"]
    pscore = rec_row.get("project_score", "")
    dist = rec_row.get("cosine_distance", "")
    #p_desc = rec_row.get("project_description", "")
    keyword_proj = normalize_str_list(rec_row.get("키워드_project", []))

    paper_list = normalize_str_list(rec_row.get("paper", []))
    patent_list = normalize_str_list(rec_row.get("patent", []))
    paper_list_count = len(paper_list)
    patent_list_count = len(patent_list)
    has_paper = len(paper_list) > 0
    has_patent = len(patent_list) > 0

    # 수행 과제/기업 파싱
    conduct_list_company = normalize_conduct_list_company(
        rec_row.get("conduct_list_company", [])
    )

    conduct_list_project = normalize_conduct_list_project(
        company_row.get("conduct_list_project", [])
    )
    has_conduct_company = len(conduct_list_company) > 0
    has_conduct_project = len(conduct_list_project) > 0

    company_region = str(company_row.get("region", "")).strip()
    region_relation = analyze_region_relation(company_region, conduct_list_company)

    # company_patent_sim 파싱
    # patent_matching_related 파싱
    patent_matching_related = company_row.get("company_patent_sim", [])
    if isinstance(patent_matching_related, str):
        try:
            patent_matching_related = ast.literal_eval(patent_matching_related)
        except:
            patent_matching_related = []

    if not isinstance(patent_matching_related, list):
        patent_matching_related = []

    valid_patent_matching_related = []
    for x in patent_matching_related:
        # dict 형태
        if isinstance(x, dict):
            title = x.get("특허명", "")
            if str(title).strip():
                valid_patent_matching_related.append({
                    "특허명": str(title).strip()
                })

        # 문자열 형태
        elif isinstance(x, str):
            title = x.strip()
            if title:
                valid_patent_matching_related.append({
                    "특허명": title
                })


    # fewshot과 같은 format 규칙
    fmt = [
        "연관성",
        "추천 과제의 우수성"
    ]

    has_similar_case = any([
        len(valid_patent_matching_related) > 0,
        has_paper,
        has_patent,
        has_conduct_company,
        has_conduct_project
    ])

    if has_similar_case:
        fmt.append("유사 사례")

    payload = {
        "company": {
            "company_id": c_id,
            "name": c_name,
            "patent_matching_related": valid_patent_matching_related,
            "has_patent_matching_related": len(valid_patent_matching_related) > 0,
            "purpose": c_purpose,
            "keyword": c_keyword,

            "region": company_region,
            "conduct_list_project": conduct_list_project,
            "has_conduct_project": has_conduct_project,

            "벤처기업여부": company_row.get("벤처기업여부", ""),
            "이노비즈여부": company_row.get("이노비즈여부", ""),
            "메인비즈여부": company_row.get("메인비즈여부", ""),
            "ASTI 여부": company_row.get("ASTI 여부", ""),
            "특구 여부": company_row.get("특구 여부", ""),

            "매출성장율_상위비율": company_row.get("매출성장율_상위비율", None),
            "매출성장율_group": extract_group(company_row.get("매출성장율_판정정보")),

            "영업이익율_상위비율": company_row.get("영업이익율_상위비율", None),
            "영업이익율_group": extract_group(company_row.get("영업이익율_판정정보")),

            "부채비율_하위비율": company_row.get("부채비율_하위비율", None),
            "부채비율_group": extract_group(company_row.get("부채비율_판정정보")),

            "연구개발비_상위비율": company_row.get("연구개발비_상위비율", None),
            "연구개발비_group": extract_group(company_row.get("연구개발비_판정정보")),

        },
        "project": {
            "project_id": pid,
            "title": pname,
            "keyword": keyword_proj,

            "총연구비_상위비율": rec_row.get("총연구비_상위비율", None),
            "총연구비_group": extract_group(rec_row.get("총연구비_판정정보")),
        },

        "conduct_list_company": conduct_list_company,
        "has_conduct_company": has_conduct_company,

        "region_relation": region_relation,

        "related_research": {
            "paper": paper_list,
            "patent": patent_list
        },
        "has_paper": has_paper,
        "has_patent": has_patent,
        "paper_list_count": paper_list_count,
        "patent_list_count": patent_list_count,

        "output_requirements": {
            "language": "ko",
            "format": fmt,
            "section_sentence_range": "각 섹션 4~6문장",
            "forbidden": ["예시", "참고", "제출", "메타"]
        }
    }
    payload = to_py(payload)

    user_prompt = (
        "아래 JSON만을 근거로 추천 근거를 작성해.\n"
        "다만 기술 설명이나 원리 설명이 필요한 경우에는 일반적인 산업 또는 기술 지식을 활용하여 설명할 수 있다.\n"
        "few-shot 예시에 나온 모든 고유명사는 예시 전용이며, 현재 출력에 절대 재사용하지 마.\n"
        "출력은 JSON 하나만 반환해. JSON 외 텍스트 금지야.\n"
        "첫 글자는 {, 마지막 글자는 } 로 끝내.\n"
        "``` 같은 코드블록은 절대 쓰지 마.\n"
        "키는 output_requirements.format에 있는 섹션 제목을 그대로 사용해.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


    messages = (
    [{"role": "system", "content": SYSTEM_PROMPT}]
    + FEWSHOT_MESSAGES
    + [{"role": "user", "content": user_prompt}]
    )

    return messages, fmt


def load_model(
    model_id="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",#"./models/Qwen3.5-35B-A3B-GPTQ-Int4",
    hf_token=None,
    tensor_parallel_size=1,
):
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    llm = LLM(
        model=model_id,
        hf_token=hf_token,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
    )

    qwen_tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token
    )

    return llm, qwen_tok


def _messages_to_prompt(messages, tokenizer):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

@torch.inference_mode()
def generate_explanation(messages, tokenizer, model,
                         expected_keys,
                         max_new_tokens=2048,
                         temperature=0.0,
                         top_p=1.0):

    prompt = _messages_to_prompt(messages, tokenizer)

    json_schema = {
        "type": "object",
        "properties": {
            k: {"type": "string"}
            for k in expected_keys
        },
        "required": expected_keys,
        "additionalProperties": False
    }

    structured_outputs = StructuredOutputsParams(
        json=json_schema
    )

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=["<|im_end|>"],
        structured_outputs=structured_outputs
    )

    outputs = model.generate([prompt], params)
    result = outputs[0].outputs[0].text.strip()

    return result



def run_generation(
    tmp,
    model,
    tokenizer,
    out_path="./result/result.csv",
    token_limit=30000,
    max_new_tokens=2048,
    overwrite=True,
):

    """
    project_id 단위로 묶어서 기업별 추천 근거를 생성하고 CSV로 저장
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and out_path.exists():
        out_path.unlink()
    file_exists = out_path.exists()


    for company_id, block in tmp.groupby("company_id", sort=False):
        company_name = block["company_name"].iloc[0]
        company_row = block.iloc[0]
        company_rows = []

        for _, rec_row in block.iterrows():
            messages, expected_keys = build_company_single_prompt(
                company_row,
                rec_row,
            )

            prompt_tokens = count_prompt_tokens(messages, tokenizer)
            if prompt_tokens > token_limit:
                continue

            raw_output = generate_explanation(
                messages,
                tokenizer=tokenizer,
                model=model,
                expected_keys=expected_keys
                )

            parsed_json = ""
            one = raw_output
            try:
                obj = json.loads(raw_output)

                cleaned_obj, is_valid = postprocess_output(obj, expected_keys)

                parsed_json = json.dumps(cleaned_obj, ensure_ascii=False)
                one = parsed_json

            except Exception as e:
                parsed_json = ""
                one = clean_forbidden_text(raw_output)
        
            company_rows.append({
                "company_id": company_id,
                "company_name": company_name,
                "project_id": rec_row.get("project_id", ""),
                "project_name": rec_row.get("project_name", ""),
                "explanation": one,
            })

        pd.DataFrame(company_rows).to_csv(
            out_path,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8-sig",
        )
        file_exists = True

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    return {
        "out_path": str(out_path)
    }

