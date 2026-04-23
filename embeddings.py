# -*- coding: utf-8 -*-
"""
向量化模块 - Embedding 生成
"""
import json
import os
import requests
import numpy as np
from typing import List
from config import CONFIG_FILE, SILICONFLOW_EMBED, OPENAI_EMBED


def load_llm_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"llm_provider": "siliconflow", "api_key": ""}


def save_llm_config(config: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ── Embedding 统一接口 ────────────────────────────────────────────────────────

def get_embedding(text: str, provider: str = "siliconflow", api_key: str = "") -> np.ndarray:
    """
    获取单条文本的embedding向量
    返回 numpy 数组
    """
    if provider == "siliconflow" or not provider:
        return _embed_siliconflow(text, api_key)
    elif provider == "openai":
        return _embed_openai(text, api_key)
    else:
        # 默认用siliconflow
        return _embed_siliconflow(text, api_key)


def get_embeddings_batch(texts: List[str], provider: str = "siliconflow",
                          api_key: str = "", show_progress: bool = False) -> List[np.ndarray]:
    """
    批量获取文本embedding（更高效）
    """
    if provider == "siliconflow" or not provider:
        return _embed_batch_siliconflow(texts, api_key, show_progress)
    elif provider == "openai":
        return _embed_batch_openai(texts, api_key)
    else:
        return _embed_batch_siliconflow(texts, api_key, show_progress)


# ── SiliconFlow BGE-M3 嵌入 ───────────────────────────────────────────────────

def _embed_siliconflow(text: str, api_key: str) -> np.ndarray:
    """调用 SiliconFlow BGE-M3 嵌入模型"""
    if not api_key:
        raise ValueError("SiliconFlow API Key 未配置，请在设置中填入")

    url = SILICONFLOW_EMBED["url"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": SILICONFLOW_EMBED["name"],
        "input": text,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    vector = data["data"][0]["embedding"]
    return np.array(vector, dtype=np.float32)


def _embed_batch_siliconflow(texts: List[str], api_key: str,
                              show_progress: bool = False) -> List[np.ndarray]:
    """
    SiliconFlow 批量嵌入（BGE-M3 支持批量）
    每批最多50条
    """
    if not api_key:
        raise ValueError("SiliconFlow API Key 未配置，请在设置中填入")

    url = SILICONFLOW_EMBED["url"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    all_vectors = []
    batch_size = 50

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "model": SILICONFLOW_EMBED["name"],
            "input": batch,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for item in data["data"]:
            all_vectors.append(np.array(item["embedding"], dtype=np.float32))

        if show_progress:
            print(f"  向量化进度：{min(i + batch_size, len(texts))}/{len(texts)}")

    return all_vectors


# ── OpenAI Embedding ──────────────────────────────────────────────────────────

def _embed_openai(text: str, api_key: str) -> np.ndarray:
    """调用 OpenAI Embedding API"""
    if not api_key:
        raise ValueError("OpenAI API Key 未配置，请在设置中填入")

    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_EMBED["name"],
        "input": text,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    vector = data["data"][0]["embedding"]
    return np.array(vector, dtype=np.float32)


def _embed_batch_openai(texts: List[str], api_key: str) -> List[np.ndarray]:
    """OpenAI 批量嵌入"""
    if not api_key:
        raise ValueError("OpenAI API Key 未配置，请在设置中填入")

    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_EMBED["name"],
        "input": texts,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return [np.array(item["embedding"], dtype=np.float32) for item in data["data"]]


# ── 向量相似度 ────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2归一化"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
