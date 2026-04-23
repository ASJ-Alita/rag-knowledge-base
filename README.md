<div align="center">

# 🏢 Training Toolkit

> 用 AI 重塑企业培训全流程

[![Training Toolkit](https://img.shields.io/badge/🧩-Training_Toolkit-2D9CDB?style=flat-square)](#)
[![RAG Engine](https://img.shields.io/badge/📚-Knowledge_Base-27AE60?style=flat-square)](#)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python)](#)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#)
[![FAISS](https://img.shields.io/badge/Vector_Store-FAISS-orange?style=flat-square)](#)

**企业知识库智能问答系统** — 培训不再只靠文档，让 AI 帮员工找到答案

上传企业培训文档（PDF/TXT/DOCX），基于 RAG 技术构建专属知识库，员工提问即可获得精准回答 + 来源追溯。

🧩 **产品矩阵** → [培训需求分析](https://github.com/ASJ-Alita/training-analyzer) · [培训效果评估](https://github.com/ASJ-Alita/kirkpatrick-eval) · [效果追踪器](https://github.com/ASJ-Alita/training-tracker) · [智能出题](https://github.com/ASJ-Alita/quiz-generator) · [培训助手](https://github.com/ASJ-Alita/training-assistant)

</div>

---

# 📚 RAG 知识库问答系统

> **Knowledge Base Q&A System — Powered by RAG**  
> 基于检索增强生成（Retrieval-Augmented Generation）的智能知识库问答系统

---

## 🎯 什么是 RAG？

RAG（检索增强生成）是一种结合**信息检索**和**大语言模型**的技术：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   用户问题   │ ──▶ │  知识检索    │ ──▶ │  LLM 生成    │
└──────────────┘     └──────────────┘     └──────────────┘
                            ↓
                     ┌──────────────┐
                     │  知识库文档  │
                     └──────────────┘
```

**优势：**
- 📖 基于真实文档回答，不会胡说八道
- 🔒 知识可控，不依赖模型训练数据
- 💡 支持私有知识库（如公司内部文档）

---

## ✨ 系统功能

| 功能 | 说明 |
|------|------|
| 📂 **多格式文档上传** | 支持 PDF / TXT / DOCX |
| 🔢 **智能文本分块** | 滑动窗口分块，保留上下文 |
| 🔍 **向量语义检索** | BGE-M3 Embedding + FAISS 索引 |
| 💬 **RAG 智能问答** | 结合检索结果生成准确回答 |
| 📌 **来源追溯** | 显示答案引用的文档来源 |
| 💾 **本地向量存储** | 索引持久化，重复使用 |

---

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 推荐 4GB+ RAM

### 安装

```bash
git clone https://github.com/ASJ-Alita/rag-knowledge-base.git
cd rag-knowledge-base

# 推荐安装（完整功能）
pip install -r requirements.txt

# 最小安装（不需要PDF/DOCX解析）
pip install streamlit requests numpy
```

### 运行

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`

---

### 使用流程

```
1. 打开程序 → 左侧边栏配置 API Key（推荐硅基流动，免费）
2. 切换到「📂 知识库管理」→ 上传文档（PDF/TXT/DOCX）
3. 等待处理完成（自动分块+向量化）
4. 切换到「💬 智能问答」→ 输入问题 → 获取回答
```

---

## ⚙️ API 配置

### 🌟 推荐：硅基流动（免费）

1. 注册：https://account.siliconflow.cn
2. 获取 API Key
3. 填入程序侧边栏

**免费额度：**
- DeepSeek-V3 模型：免费使用
- BGE-M3 嵌入模型：免费使用

### 其他平台

| 平台 | 模型 | 说明 |
|------|------|------|
| 硅基流动 | DeepSeek-V3 + BGE-M3 | 🌟 免费推荐 |
| OpenAI | GPT-4o-mini + ada-002 | 需科学上网 |
| 智谱AI | GLM-4-flash | 免费额度 |

---

## 📁 项目结构

```
rag-knowledge-base/
├── app.py                  # Streamlit Web界面
├── rag_engine.py            # RAG核心引擎（检索+生成）
├── document_processor.py    # 文档解析（PDF/TXT/DOCX）
├── embeddings.py            # 向量化模块
├── vector_store.py          # FAISS向量数据库
├── config.py                # 配置参数
├── documents/               # 上传的文档（自动创建）
├── indexes/                 # 向量索引（自动创建）
├── requirements.txt
└── README.md
```

---

## 🔬 技术架构

```
文档上传
   ↓
extract_text_from_file()   ← 支持 PDF/TXT/DOCX
   ↓
chunk_text()               ← 滑动窗口分块（400字/块，80字重叠）
   ↓
get_embeddings_batch()     ← BGE-M3 向量化（SiliconFlow API）
   ↓
VectorStore.add_chunks()    ← 存入 FAISS 索引
   ↓
用户提问
   ↓
RAGEngine.ask()            ← 检索相关块 → LLM生成回答
   ↓
返回回答 + 引用来源
```

---

## 🎯 面试亮点

| 能力维度 | 体现点 |
|----------|--------|
| **RAG 技术深度** | 完整的 RAG 流程：分块→向量化→检索→生成 |
| **向量数据库** | FAISS 索引原理与使用 |
| **大模型应用** | 多平台 API 调用、Prompt 工程 |
| **文档处理** | PDF/TXT/DOCX 多格式解析 |
| **系统架构** | 模块化设计，清晰的数据流 |
| **Web 开发** | Streamlit 快速构建数据应用 |

---

## 🔮 进阶方向

- [ ] 支持网页 URL 抓取入库
- [ ] 多文档对比问答
- [ ] 支持图片/表格理解
- [ ] 流式输出（实时显示回答）
- [ ] 对话历史记忆
- [ ] 企业微信/钉钉集成

---

## 📌 关于作者

**培训技术专家** | 12年IT教育经验 | AI应用爱好者  
GitHub: https://github.com/ASJ-Alita
