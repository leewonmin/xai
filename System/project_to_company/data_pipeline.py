import ast
from pathlib import Path
import json

import numpy as np
import pandas as pd
from numpy.linalg import norm
import torch
from transformers import AutoTokenizer, AutoModel


def load_data(data_path="./data/data.csv"):
    return pd.read_csv(data_path)

# 사용자에게 입력받은 과제명의 해당 데이터 추출
def select_project_matches(df, project_name, top_n):
    target_name = str(project_name).replace(" ", "").strip()

    df = df.copy()
    df["_project_name_norm"] = (
        df["project_name"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )

    matched = (
        df[
            df["_project_name_norm"].str.contains(
                target_name,
                na=False,
            )
        ]
        .drop(columns=["_project_name_norm"])
        .head(top_n)
        .copy()
    )

    if matched.empty:
        raise ValueError(f"'{project_name}'에 해당하는 데이터가 없습니다.")

    return matched


# None, NaN, "none"/"nan"/"null" 값을 빈 문자열로 정리
def clean_text(x):
    if x is None:
        return ""

    if isinstance(x, float) and pd.isna(x):
        return ""

    s = str(x).strip()

    if s.lower() in {"none", "nan", "null"}:
        return ""

    return s


# 문자열로 저장된 JSON/list/dict 구조를 실제 Python 객체로 복구
def parse_struct(x):
    if x is None:
        return None

    if isinstance(x, float) and pd.isna(x):
        return None

    if isinstance(x, (list, dict)):
        return x

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, str):
        x = x.strip()

        if not x:
            return None

        try:
            return json.loads(x)
        except Exception:
            pass

        try:
            return ast.literal_eval(x)
        except Exception:
            return x

    return x


# 입력값을 리스트로 정규화 (keep_scaler=False: 리스트가 아닌값은 빈 리스트로, string_only=True: 리스트 안의 문자열 값이 아닌값 삭제)
def to_list(x, keep_scalar=True, string_only=False):
    x = parse_struct(x)

    if x is None:
        return []

    if isinstance(x, list):
        values = x
    else:
        if keep_scalar:
            values = [x]
        else:
            return []

    if string_only:
        return [
            clean_text(v)
            for v in values
            if clean_text(v)
        ]

    return values

# 수행 기업/과제, 기업의 특허명 추출 시 유사도 계산
def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1)

#임베딩
def embed_texts(texts, model, tokenizer, device):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)

    embeddings = mean_pooling(
        outputs.last_hidden_state,
        encoded["attention_mask"],
    )

    embeddings = torch.nn.functional.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    return embeddings.cpu().numpy()

# 수행 기업/과제 유사도 높은 항목만 필터링
def filter_conduct_list(row_embed, conduct_list, threshold=0.5, top_n=3, dedup_key=None):
    conduct = to_list(conduct_list, keep_scalar=True)
    if not conduct:
        return None

    embed = to_list(row_embed, keep_scalar=True)
    if not embed:
        return None

    result = []

    for item in conduct:
        if not isinstance(item, dict):
            continue

        item_embed = item.get("embedding")
        if item_embed is None:
            continue

        item_embed = to_list(item_embed, keep_scalar=True)
        if not item_embed:
            continue

        try:
            sim = cosine_similarity(embed, item_embed)
        except Exception:
            continue

        if sim >= threshold:
            result.append((sim, item))

    if not result:
        return None

    result = sorted(result, key=lambda x: x[0], reverse=True)

    if dedup_key is not None:
        seen = set()
        deduped = []

        for sim, item in result:
            val = item.get(dedup_key)

            if val in seen:
                continue

            seen.add(val)
            deduped.append((sim, item))

        result = deduped

    result = [item for _, item in result[:top_n]]
    return result if result else None

# 수행 기업/과제 에서 유사도 기반 필터링 후 embedding 값 제거(llm 입력 토큰 줄이기 위함)
def remove_embedding(x):
    x = to_list(x, keep_scalar=True)
    cleaned = []

    for item in x:
        if isinstance(item, dict):
            item = item.copy()
            item.pop("embedding", None)

        cleaned.append(item)

    return cleaned if cleaned else None

# 기업의 특허 리스트에서 중복 값 제거
def remove_duplicate_patents(x):
    if not isinstance(x, list):
        return x

    seen = set()
    unique = []

    for p in x:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique

# 해당 과제와 매칭된 기업의 특허를 유사도 기반 추출 
def build_company_patent_sim(
    company_patent,
    project_embed,
    model,
    tokenizer,
    device,
    threshold=0.4,
    top_n=3,
):
    patents = to_list(company_patent, keep_scalar=True)
    patents = remove_duplicate_patents(patents)

    project_embed = to_list(project_embed, keep_scalar=True)

    if not patents or not project_embed:
        return None

    patent_texts = []

    for p in patents:
        text = str(p).strip()

        if not text or text.lower() in {"nan", "none", "null"}:
            continue

        patent_texts.append(text)

    if not patent_texts:
        return None

    
    patent_embeds = embed_texts(
        patent_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    result = []

    for patent_text, patent_embed in zip(patent_texts, patent_embeds):
        try:
            sim = cosine_similarity(project_embed, patent_embed)
        except Exception:
            continue

        if sim >= threshold:
            result.append((sim, patent_text))

    if not result:
        return None

    result = sorted(result, key=lambda x: x[0], reverse=True)

    output = [
        patent_text
        for sim, patent_text in result[:top_n]
    ]

    return output if output else None

#유사도 기반으로 추출된 기업의 특허명 컬럼 생성
def add_company_patent_sim(
    tmp,
    model,
    tokenizer,
    device,
):
    if "company_patent" not in tmp.columns:
        return tmp

    if "project_norm_embed" not in tmp.columns:
        return tmp

    tmp["company_patent_sim"] = tmp.apply(
        lambda row: build_company_patent_sim(
            company_patent=row["company_patent"],
            project_embed=row["project_norm_embed"],
            model=model,
            tokenizer=tokenizer,
            device=device,
            threshold=0.4,
            top_n=3,
        ),
        axis=1,
    )

    return tmp

# 임베딩 모델 로딩
def load_embedding_model(model_name="BAAI/bge-m3"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return model, tokenizer, device


def preprocess(
    tmp,
    embed_model=None,
    embed_tokenizer=None,
    embed_device=None,
):
    tmp = tmp.copy()

    if "project_name" not in tmp.columns and "과제명" in tmp.columns:
        tmp["project_name"] = tmp["과제명"]

    if "conduct_list_project" in tmp.columns and "company_norm_embed" in tmp.columns:
        tmp["conduct_list_project"] = tmp.apply(
            lambda row: filter_conduct_list(
                row["company_norm_embed"],
                row["conduct_list_project"],
                threshold=0.5,
                top_n=3,
                dedup_key="project_id",
            ),
            axis=1,
        )

    if "conduct_list_company" in tmp.columns and "project_norm_embed" in tmp.columns:
        tmp["conduct_list_company"] = tmp.apply(
            lambda row: filter_conduct_list(
                row["project_norm_embed"],
                row["conduct_list_company"],
                threshold=0.5,
                top_n=3,
                dedup_key="company_id",
            ),
            axis=1,
        )

    for col in ["conduct_list_company", "conduct_list_project"]:
        if col in tmp.columns:
            tmp[col] = tmp[col].apply(remove_embedding)

    if "company_patent" in tmp.columns:
        tmp["company_patent"] = (
            tmp["company_patent"]
            .apply(lambda x: to_list(x, keep_scalar=True))
            .apply(remove_duplicate_patents)
        )

    if embed_model is not None and embed_tokenizer is not None and embed_device is not None:
        tmp = add_company_patent_sim(
            tmp,
            model=embed_model,
            tokenizer=embed_tokenizer,
            device=embed_device,
        )

    if "patent" in tmp.columns:
        tmp["patent"] = (
            tmp["patent"]
            .apply(lambda x: to_list(x, keep_scalar=True))
            .apply(remove_duplicate_patents)
        )

    tmp = tmp.drop(
        columns=["company_norm_embed", "project_norm_embed"],
        errors="ignore",
    )

    return tmp
