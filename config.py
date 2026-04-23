# -*- coding: utf-8 -*-
"""
配置模块
"""
import os

# ── 项目路径 ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
CONFIG_FILE = os.path.join(BASE_DIR, "api_config.json")

# ── 向量化配置 ─────────────────────────────────────────────────────────────────

# SiliconFlow 嵌入模型（免费）
SILICONFLOW_EMBED = {
    "name": "BAAI/bge-m3",
    "url": "https://api.siliconflow.cn/v1/embeddings",
    "dimension": 1024,  # bge-m3 输出维度
}

# OpenAI 嵌入模型（备用）
OPENAI_EMBED = {
    "name": "text-embedding-3-small",
    "dimension": 1536,
}

# ── LLM 配置 ──────────────────────────────────────────────────────────────────

LLM_PROVIDERS = {
    "siliconflow": {
        "name": "硅基流动 DeepSeek-V3（推荐）",
        "chat_url": "https://api.siliconflow.cn/v1/chat/completions",
        "model": "deepseek-ai/DeepSeek-V3",
        "temperature": 0.3,
        "max_tokens": 2000,
    },
    "openai": {
        "name": "OpenAI GPT-4o-mini",
        "chat_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 2000,
    },
    "zhihui": {
        "name": "智谱 GLM-4-flash",
        "chat_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "model": "glm-4-flash",
        "temperature": 0.3,
        "max_tokens": 2000,
    },
}

# ── RAG 配置 ──────────────────────────────────────────────────────────────────

CHUNK_SIZE = 400       # 每块字符数
CHUNK_OVERLAP = 80    # 重叠字符数
TOP_K = 5              # 检索相关块数
SIMILARITY_THRESHOLD = 0.3  # 相似度阈值
