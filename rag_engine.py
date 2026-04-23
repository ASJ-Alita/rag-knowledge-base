# -*- coding: utf-8 -*-
"""
RAG 问答引擎
Retrieval-Augmented Generation
"""
import os
import json
import requests
import time
from typing import List, Optional
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, SIMILARITY_THRESHOLD,
    CONFIG_FILE, LLM_PROVIDERS, SILICONFLOW_EMBED
)
from document_processor import extract_text_from_file, chunk_text, get_file_info
from embeddings import get_embeddings_batch, load_llm_config
from vector_store import VectorStore


class RAGEngine:
    """
    RAG 问答引擎
    核心流程：
    1. 上传文档 → 解析文本 → 分块 → 向量化 → 存储
    2. 用户提问 → 向量化查询 → 检索相关块 → LLM生成回答
    """

    def __init__(self):
        self.config = load_llm_config()
        self.llm_provider = self.config.get("llm_provider", "siliconflow")
        self.api_key = self.config.get("api_key", "")

        # 向量维度（BGE-M3=1024, OpenAI=1536）
        dim = (SILICONFLOW_EMBED["dimension"] if self.llm_provider == "siliconflow"
               else 1536)
        self.vector_store = VectorStore(dimension=dim, provider=self.llm_provider)

        self.conversation_history = []  # 对话历史

    # ── 文档处理 ──────────────────────────────────────────────────────────────

    def ingest_document(self, filepath: str,
                        chunk_size: int = CHUNK_SIZE,
                        chunk_overlap: int = CHUNK_OVERLAP) -> dict:
        """
        处理并入库一个文档
        返回处理结果统计
        """
        print(f"📄 正在处理文档：{os.path.basename(filepath)}")

        # 1. 提取文本
        text = extract_text_from_file(filepath)
        if not text.strip():
            raise ValueError("文档内容为空")
        print(f"   提取文本：{len(text)} 字")

        # 2. 分块
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        print(f"   切分文本块：{len(chunks)} 块")

        # 3. 获取元数据
        file_info = get_file_info(filepath)
        source_name = file_info["name"]

        # 4. 向量化
        print(f"   正在向量化（{len(chunks)} 个文本块）...")
        texts = [c["text"] for c in chunks]
        embeddings = get_embeddings_batch(texts, provider=self.llm_provider,
                                          api_key=self.api_key, show_progress=True)

        # 5. 构建元数据
        chunk_metas = []
        for i, c in enumerate(chunks):
            chunk_metas.append({
                "text": c["text"],
                "source": source_name,
                "chunk_id": i,
                "total_chars": len(c["text"]),
                "ingest_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        # 6. 存入向量数据库
        self.vector_store.add_chunks(chunk_metas, embeddings)
        self.vector_store.save()

        return {
            "filename": source_name,
            "chars": len(text),
            "chunks": len(chunks),
            "status": "success",
        }

    # ── RAG 问答 ─────────────────────────────────────────────────────────────

    def ask(self, question: str, top_k: int = TOP_K,
            threshold: float = SIMILARITY_THRESHOLD,
            stream: bool = False) -> dict:
        """
        RAG 问答
        返回包含回答和引用来源的字典
        """
        if not question.strip():
            raise ValueError("问题不能为空")

        if not self.api_key:
            raise ValueError("API Key 未配置，请在设置中配置")

        # 1. 将问题向量化
        print(f"🔍 正在检索相关内容...")
        q_embedding = get_embeddings_batch([question], provider=self.llm_provider,
                                           api_key=self.api_key)[0]

        # 2. 检索相关文本块
        results = self.vector_store.search(q_embedding, top_k=top_k, threshold=threshold)

        if not results:
            return {
                "answer": "⚠️ 在知识库中未找到与您问题相关的内容。\n\n"
                          "可能的原因：\n"
                          "• 知识库为空（请先上传文档）\n"
                          "• 问题与文档内容不相关\n"
                          "• 文档内容不够详细",
                "sources": [],
                "question": question,
            }

        # 3. 构建上下文
        context_parts = []
        for i, (chunk, score) in enumerate(results, 1):
            context_parts.append(
                f"【来源{i}】{chunk['source']}（相关度：{score:.2f}）\n{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

        # 4. 构建Prompt
        system_prompt = (
            "你是一位专业的知识库问答助手，名为「智库助手」。\n"
            "你的职责是：基于提供的知识库内容，准确回答用户的问题。\n\n"
            "重要规则：\n"
            "1. 只根据【参考资料】中的内容回答，不要编造信息\n"
            "2. 如果知识库内容不足以完整回答，请明确说明\n"
            "3. 回答要专业、准确、简洁\n"
            "4. 可以引用【来源】标注内容出处\n"
            "5. 用中文回答"
        )

        user_prompt = (
            f"【用户问题】\n{question}\n\n"
            f"【参考资料】\n{context}\n\n"
            f"请根据参考资料回答用户问题。如果资料不足，请说明。"
        )

        # 5. 调用LLM生成回答
        answer = self._call_llm(system_prompt, user_prompt, stream=stream)

        return {
            "answer": answer,
            "sources": [
                {"text": chunk["text"][:200] + "...",
                 "source": chunk["source"],
                 "score": score}
                for chunk, score in results
            ],
            "question": question,
            "retrieved_count": len(results),
        }

    def _call_llm(self, system_prompt: str, user_prompt: str, stream: bool = False) -> str:
        """调用大模型"""
        cfg = LLM_PROVIDERS.get(self.llm_provider, LLM_PROVIDERS["siliconflow"])

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
        }

        if self.llm_provider == "siliconflow":
            url = cfg["chat_url"]
        elif self.llm_provider == "zhihui":
            url = cfg["chat_url"]
        elif self.llm_provider == "openai":
            url = cfg["chat_url"]
        else:
            url = cfg["chat_url"]

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """获取知识库统计"""
        vs_stats = self.vector_store.get_stats()
        return {
            **vs_stats,
            "llm_provider": self.llm_provider,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }

    def clear_knowledge_base(self):
        """清空知识库"""
        self.vector_store.clear()
        self.conversation_history = []

    def reload_config(self):
        """重新加载配置"""
        self.config = load_llm_config()
        self.llm_provider = self.config.get("llm_provider", "siliconflow")
        self.api_key = self.config.get("api_key", "")
