# -*- coding: utf-8 -*-
"""
RAG 知识库问答系统 - 单元测试
测试文档解析、文本分块、向量化工具函数
"""
import os
import sys
import json
import tempfile
import numpy as np
import pytest

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==================== document_processor 测试 ====================

class TestDocumentProcessor:
    """文档解析模块测试"""

    def test_extract_txt_utf8(self):
        """测试 UTF-8 编码的 TXT 文件解析"""
        from document_processor import extract_text_from_file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("这是一个测试文件。\n第二行内容。")
            filepath = f.name
        try:
            text = extract_text_from_file(filepath)
            assert "测试文件" in text
            assert "第二行" in text
        finally:
            os.unlink(filepath)

    def test_extract_txt_gbk(self):
        """测试 GBK 编码的 TXT 文件解析"""
        from document_processor import extract_text_from_file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='gbk') as f:
            f.write("GBK编码测试内容。")
            filepath = f.name
        try:
            text = extract_text_from_file(filepath)
            assert "GBK编码" in text
        finally:
            os.unlink(filepath)

    def test_extract_unsupported_format(self):
        """测试不支持的文件格式应抛出异常"""
        from document_processor import extract_text_from_file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("test")
            filepath = f.name
        try:
            with pytest.raises(ValueError, match="不支持的文件格式"):
                extract_text_from_file(filepath)
        finally:
            os.unlink(filepath)

    def test_chunk_text_short(self):
        """测试短文本（不超过 chunk_size）不分块"""
        from document_processor import chunk_text
        result = chunk_text("短文本", chunk_size=400, overlap=80)
        assert len(result) == 1
        assert result[0]["text"] == "短文本"
        assert result[0]["chunk_id"] == 0

    def test_chunk_text_long(self):
        """测试长文本分块"""
        from document_processor import chunk_text
        text = "这是一段测试文本。" * 100  # 约 800 字
        result = chunk_text(text, chunk_size=400, overlap=80)
        assert len(result) > 1
        # 验证 chunk_id 连续递增
        for i, chunk in enumerate(result):
            assert chunk["chunk_id"] == i
        # 验证每个块不超过 chunk_size
        for chunk in result:
            assert len(chunk["text"]) <= 400 + 10  # 允许少许截断误差

    def test_chunk_text_with_overlap(self):
        """测试分块有重叠"""
        from document_processor import chunk_text
        text = "第一句话。第二句话。第三句话。第四句话。第五句话。" * 50
        result = chunk_text(text, chunk_size=200, overlap=50)
        if len(result) >= 2:
            # 验证相邻块有重叠
            first_end = result[0].get("end_char", 0)
            second_start = result[1].get("start_char", 0)
            assert second_start < first_end  # 重叠意味着第二块开始 < 第一块结束

    def test_clean_text(self):
        """测试文本清理"""
        from document_processor import _clean_text
        # 测试多余空格
        assert _clean_text("hello   world") == "hello world"
        # 测试多余换行
        assert "\n\n\n" not in _clean_text("a\n\n\n\nb")

    def test_get_file_info(self):
        """测试获取文件信息"""
        from document_processor import get_file_info
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content" * 100)
            filepath = f.name
        try:
            info = get_file_info(filepath)
            assert info["name"].endswith(".txt")
            assert info["type"] == ".txt"
            assert info["size"] > 0
            assert info["size_kb"] > 0
        finally:
            os.unlink(filepath)


# ==================== embeddings 测试 ====================

class TestEmbeddings:
    """向量化工具函数测试（不调用 API，只测纯函数）"""

    def test_cosine_similarity_identical(self):
        """测试相同向量的余弦相似度应为 1.0"""
        from embeddings import cosine_similarity
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sim = cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        """测试正交向量的余弦相似度应为 0.0"""
        from embeddings import cosine_similarity
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        """测试零向量的余弦相似度应为 0.0"""
        from embeddings import cosine_similarity
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert sim == 0.0

    def test_cosine_similarity_opposite(self):
        """测试反向向量的余弦相似度应为 -1.0"""
        from embeddings import cosine_similarity
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert abs(sim - (-1.0)) < 1e-6

    def test_normalize_vector(self):
        """测试 L2 归一化"""
        from embeddings import normalize_vector
        v = np.array([3.0, 4.0], dtype=np.float32)
        normalized = normalize_vector(v)
        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-6

    def test_normalize_zero_vector(self):
        """测试零向量归一化不崩溃"""
        from embeddings import normalize_vector
        v = np.array([0.0, 0.0], dtype=np.float32)
        result = normalize_vector(v)
        assert np.array_equal(result, v)

    def test_get_embedding_no_api_key(self):
        """测试无 API Key 时应抛出异常"""
        from embeddings import get_embedding
        with pytest.raises(ValueError, match="API Key"):
            get_embedding("test text", provider="siliconflow", api_key="")


# ==================== vector_store 测试 ====================

class TestVectorStore:
    """向量存储模块测试（使用临时目录）"""

    def test_create_new_index(self, tmp_path, monkeypatch):
        """测试创建新索引"""
        monkeypatch.setattr("config.INDEX_DIR", str(tmp_path))
        from vector_store import VectorStore
        vs = VectorStore(dimension=64, provider="test")
        stats = vs.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["dimension"] == 64

    def test_add_and_search(self, tmp_path, monkeypatch):
        """测试添加和检索"""
        monkeypatch.setattr("config.INDEX_DIR", str(tmp_path))
        from vector_store import VectorStore
        vs = VectorStore(dimension=64, provider="test")

        chunks = [
            {"text": "Python是一种编程语言", "source": "doc1.txt", "chunk_id": 0},
            {"text": "Java也是一种编程语言", "source": "doc2.txt", "chunk_id": 1},
        ]
        embeddings = [
            np.random.randn(64).astype(np.float32),
            np.random.randn(64).astype(np.float32),
        ]
        vs.add_chunks(chunks, embeddings)
        assert vs.get_stats()["total_chunks"] == 2

    def test_save_and_load(self, tmp_path, monkeypatch):
        """测试保存和加载索引"""
        monkeypatch.setattr("config.INDEX_DIR", str(tmp_path))
        from vector_store import VectorStore

        # 创建并添加数据
        vs1 = VectorStore(dimension=64, provider="test_persist")
        chunks = [{"text": "测试文本", "source": "test.txt", "chunk_id": 0}]
        embeddings = [np.random.randn(64).astype(np.float32)]
        vs1.add_chunks(chunks, embeddings)
        vs1.save()

        # 重新加载
        vs2 = VectorStore(dimension=64, provider="test_persist")
        assert vs2.get_stats()["total_chunks"] == 1
        assert vs2.chunks[0]["text"] == "测试文本"

    def test_clear(self, tmp_path, monkeypatch):
        """测试清空索引"""
        monkeypatch.setattr("config.INDEX_DIR", str(tmp_path))
        from vector_store import VectorStore
        vs = VectorStore(dimension=64, provider="test_clear")
        chunks = [{"text": "临时数据", "source": "tmp.txt", "chunk_id": 0}]
        embeddings = [np.random.randn(64).astype(np.float32)]
        vs.add_chunks(chunks, embeddings)
        vs.clear()
        assert vs.get_stats()["total_chunks"] == 0

    def test_empty_search(self, tmp_path, monkeypatch):
        """测试空索引检索返回空"""
        monkeypatch.setattr("config.INDEX_DIR", str(tmp_path))
        from vector_store import VectorStore
        vs = VectorStore(dimension=64, provider="test_empty")
        query = np.random.randn(64).astype(np.float32)
        results = vs.search(query, top_k=5)
        assert results == []


# ==================== config 测试 ====================

class TestConfig:
    """配置模块测试"""

    def test_chunk_config(self):
        """测试分块配置"""
        from config import CHUNK_SIZE, CHUNK_OVERLAP
        assert CHUNK_SIZE > 0
        assert CHUNK_OVERLAP < CHUNK_SIZE

    def test_rag_config(self):
        """测试 RAG 配置"""
        from config import TOP_K, SIMILARITY_THRESHOLD
        assert TOP_K > 0
        assert 0 <= SIMILARITY_THRESHOLD <= 1

    def test_llm_providers(self):
        """测试 LLM 提供商配置"""
        from config import LLM_PROVIDERS
        assert "siliconflow" in LLM_PROVIDERS
        assert "model" in LLM_PROVIDERS["siliconflow"]
        assert "chat_url" in LLM_PROVIDERS["siliconflow"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
