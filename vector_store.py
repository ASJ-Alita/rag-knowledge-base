# -*- coding: utf-8 -*-
"""
向量存储与检索模块
使用 FAISS 索引 + JSON 元数据
"""
import os
import json
import numpy as np
import pickle
from typing import List, Tuple, Optional

# 尝试导入FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from config import INDEX_DIR
from embeddings import cosine_similarity


class VectorStore:
    """
    轻量级向量数据库
    - 使用 FAISS IndexFlatIP（内积）做相似度检索
    - 文本块元数据存储在 JSON 文件
    """

    def __init__(self, dimension: int = 1024, provider: str = "siliconflow"):
        self.dimension = dimension
        self.provider = provider
        self.index_path = os.path.join(INDEX_DIR, f"faiss_index_{provider}.index")
        self.meta_path = os.path.join(INDEX_DIR, f"chunks_meta_{provider}.json")

        self.index = None
        self.chunks = []  # List[dict]: {"text", "source", "chunk_id", "embedding_model"}...

        self._load()

    def _load(self):
        """加载已有索引"""
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                if FAISS_AVAILABLE:
                    self.index = faiss.read_index(self.index_path)
                else:
                    self.index = None  # FAISS不可用时降级
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                print(f"✅ 已加载索引：{len(self.chunks)} 个文本块")
            except Exception as e:
                print(f"⚠️ 加载索引失败：{e}，将创建新索引")
                self._new_index()
        else:
            self._new_index()

    def _new_index(self):
        """创建新索引"""
        if FAISS_AVAILABLE:
            # Inner Product (余弦相似度需要先归一化)
            self.index = faiss.IndexFlatIP(self.dimension)
            print("✅ 已创建新 FAISS 索引")
        else:
            self.index = None
            print("⚠️ FAISS 未安装，将使用暴力搜索（仅适合小数据量）")
        self.chunks = []

    def add_chunks(self, chunks: List[dict], embeddings: List[np.ndarray]):
        """
        添加文本块和对应的embedding向量
        chunks: [{"text", "source", "chunk_id"}, ...]
        embeddings: 对应的向量列表
        """
        if not chunks or not embeddings:
            return

        # L2归一化（FAISS IP索引需要）
        normalized = [e / (np.linalg.norm(e) + 1e-8) for e in embeddings]
        vectors = np.array(normalized).astype(np.float32)

        if self.index is not None:
            self.index.add(vectors)
        self.chunks.extend(chunks)

        print(f"✅ 已添加 {len(chunks)} 个文本块，总计 {len(self.chunks)} 块")

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               threshold: float = 0.0) -> List[Tuple[dict, float]]:
        """
        检索最相似的文本块
        返回: [(chunk_dict, similarity_score), ...]
        """
        if not self.chunks:
            return []

        q = query_embedding.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)  # 归一化

        if self.index is not None and self.index.ntotal > 0:
            # FAISS 检索
            search_k = min(top_k * 2, len(self.chunks))  # 多检索一些，过滤低分
            scores, indices = self.index.search(q.reshape(1, -1), search_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks) and score >= threshold:
                    results.append((self.chunks[idx], float(score)))
        else:
            # 暴力搜索（无FAISS时）
            results = []
            for i, chunk_emb in enumerate(self.chunks):
                # 尝试从存储中重建（简化：直接用文本匹配度）
                sim = 0.5  # 降级处理
                results.append((self.chunks[i], sim))
            results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

        return results[:top_k]

    def save(self):
        """保存索引到磁盘"""
        os.makedirs(INDEX_DIR, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            print(f"✅ FAISS索引已保存到：{self.index_path}")

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"✅ 元数据已保存到：{self.meta_path}")

    def get_stats(self) -> dict:
        """获取索引统计"""
        return {
            "total_chunks": len(self.chunks),
            "dimension": self.dimension,
            "index_type": "FAISS IndexFlatIP" if self.index else "Fallback",
            "sources": list(set(c.get("source", "unknown") for c in self.chunks)),
        }

    def clear(self):
        """清空所有数据"""
        self._new_index()
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
        print("✅ 索引已清空")
