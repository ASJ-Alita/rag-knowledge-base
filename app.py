# -*- coding: utf-8 -*-
"""
RAG 知识库问答系统 - Streamlit Web界面
Knowledge Base Q&A System
"""
import streamlit as st
import os
import json
import time
import pandas as pd
from pathlib import Path

# 导入本地模块
from config import DOCS_DIR, CONFIG_FILE, LLM_PROVIDERS, SILICONFLOW_EMBED
from document_processor import extract_text_from_file, get_file_info
from rag_engine import RAGEngine
from embeddings import load_llm_config, save_llm_config, SILICONFLOW_EMBED

# ── 页面配置 ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📚 RAG知识库问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义样式 ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: #f0f2f5; }
.css-1d391kg { padding-top: 1rem; }
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
}
.stTextInput>div>div>input, .stTextArea>div>div>textarea {
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}
.success-box {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border-radius: 12px; padding: 16px 20px;
    border-left: 4px solid #059669;
    margin: 8px 0;
}
.info-box {
    background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    border-radius: 12px; padding: 16px 20px;
    border-left: 4px solid #3b82f6;
    margin: 8px 0;
}
.chat-user {
    background: #4f46e5; color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px; margin: 8px 0;
    max-width: 75%; margin-left: auto;
}
.chat-assistant {
    background: white; color: #1e293b;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px; margin: 8px 0;
    border: 1px solid #e2e8f0;
    max-width: 75%;
}
.source-card {
    background: #f8fafc; border-radius: 8px;
    padding: 10px 14px; margin: 6px 0;
    border-left: 3px solid #94a3b8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ── 会话状态初始化 ────────────────────────────────────────────────────────────
def init_session():
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = RAGEngine()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = []
    # 加载已有文件名
    stats = st.session_state.rag_engine.get_stats()
    if stats.get("sources"):
        st.session_state.ingested_files = stats["sources"]


init_session()


# ── 侧边栏：设置 & 知识库管理 ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ API 设置")

    config = load_llm_config()
    provider = st.selectbox(
        "选择大模型",
        options=list(LLM_PROVIDERS.keys()),
        format_func=lambda x: LLM_PROVIDERS[x]["name"],
        index=list(LLM_PROVIDERS.keys()).index(config.get("llm_provider", "siliconflow"))
    )
    api_key = st.text_input(
        "API Key",
        value=config.get("api_key", ""),
        type="password",
        help="在对应平台获取API Key"
    )

    hints = {
        "siliconflow": "🌐 注册：account.siliconflow.cn\nDeepSeek-V3 模型免费使用",
        "openai": "🤖 platform.openai.com\n需要科学上网",
        "zhihui": "🧠 open.bigmodel.cn\nGLM-4-flash 免费额度",
    }
    st.caption(hints.get(provider, ""))

    if st.button("💾 保存配置", use_container_width=True, type="primary"):
        save_llm_config({"llm_provider": provider, "api_key": api_key})
        st.session_state.rag_engine.reload_config()
        st.success("✅ 配置已保存！")

    st.divider()

    # 知识库统计
    st.markdown("## 📊 知识库状态")
    stats = st.session_state.rag_engine.get_stats()
    col1, col2 = st.columns(2)
    col1.metric("文档数", len(st.session_state.ingested_files))
    col2.metric("文本块", stats.get("total_chunks", 0))

    if st.session_state.ingested_files:
        st.markdown("**📄 已入库文档：**")
        for f in st.session_state.ingested_files:
            st.markdown(f"- `{f}`")

    st.divider()

    # 清空知识库
    if st.button("🗑️ 清空知识库", use_container_width=True, type="secondary"):
        st.session_state.rag_engine.clear_knowledge_base()
        st.session_state.ingested_files = []
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("💡 上传文档后即可开始问答")


# ── 主界面 ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding: 8px 0 16px">
    <h1 style="color:#1e293b; margin:0">📚 RAG 知识库问答系统</h1>
    <p style="color:#64748b; margin:4px 0 0">Retrieval-Augmented Generation · 基于检索增强的智能问答</p>
</div>
""", unsafe_allow_html=True)

# ── Tab 1: 知识库管理 ───────────────────────────────────────────────────────
tab_kb, tab_qa = st.tabs(["📂 知识库管理", "💬 智能问答"])

with tab_kb:
    st.markdown("### 📤 上传文档到知识库")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **支持的格式：** `.txt` / `.pdf` / `.docx`

        **处理流程：**
        1. 上传文件 → 文本提取 → 智能分块 → 向量化存储
        2. 文档入库后，即可在问答中使用
        """)
    with col2:
        st.markdown("""
        **分块设置：**
        - 块大小：400字符
        - 重叠：80字符
        - 向量模型：BGE-M3（1024维）

        💡 建议单文档不超过50页/10万字
        """)

    uploaded_files = st.file_uploader(
        "选择文件上传",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="支持多文件批量上传"
    )

    if uploaded_files:
        st.markdown(f"**已选择 {len(uploaded_files)} 个文件：**")

        progress_bar = st.progress(0, text="准备上传...")
        for i, uploaded_file in enumerate(uploaded_files):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{uploaded_file.name}**  ({uploaded_file.size / 1024:.1f} KB)")
            with col2:
                st.markdown(f"`{uploaded_file.type}`")

            # 保存文件
            os.makedirs(DOCS_DIR, exist_ok=True)
            save_path = os.path.join(DOCS_DIR, uploaded_file.name)

            # 检测重名
            if os.path.exists(save_path):
                base, ext = os.path.splitext(uploaded_file.name)
                save_path = os.path.join(DOCS_DIR, f"{base}_{int(time.time())}{ext}")

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with col3:
                with st.spinner(f"处理中..."):
                    try:
                        result = st.session_state.rag_engine.ingest_document(save_path)
                        if result["filename"] not in st.session_state.ingested_files:
                            st.session_state.ingested_files.append(result["filename"])
                        st.success(f"✅ 入库成功！{result['chunks']}块")
                    except Exception as e:
                        st.error(f"❌ 失败：{str(e)[:60]}")

            progress_bar.progress((i + 1) / len(uploaded_files), text=f"已完成 {i+1}/{len(uploaded_files)}")

        st.rerun()

    # 知识库概览
    st.divider()
    st.markdown("### 📋 知识库概览")

    stats = st.session_state.rag_engine.get_stats()
    if stats.get("total_chunks", 0) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("文本块总数", stats["total_chunks"])
        c2.metric("文档数", len(st.session_state.ingested_files))
        c3.metric("向量维度", stats["dimension"])
        c4.metric("向量模型", "BGE-M3")

        st.markdown("**入库文档列表：**")
        for f in st.session_state.ingested_files:
            st.markdown(f"- `{f}`")
    else:
        st.info("📭 知识库为空，请上传文档后开始问答")


# ── Tab 2: 智能问答 ──────────────────────────────────────────────────────────
with tab_qa:
    st.markdown("### 💬 智能问答")

    # 检查配置
    config = load_llm_config()
    if not config.get("api_key"):
        st.warning("⚠️ 请先在左侧设置中配置 API Key")

    # 显示对话历史
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-assistant">{msg["content"]}</div>', unsafe_allow_html=True)
                if msg.get("sources"):
                    with st.expander("📌 查看参考来源", expanded=False):
                        for j, s in enumerate(msg["sources"], 1):
                            score_bar = "█" * int(s["score"] * 20) + "░" * (20 - int(s["score"] * 20))
                            st.markdown(
                                f'<div class="source-card">'
                                f'<b>来源{j}</b>：{s["source"]} &nbsp;|&nbsp; 相似度：{s["score"]:.2f} {score_bar}'
                                f'<br><span style="color:#64748b">{s["text"]}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

    # 问题输入
    st.divider()
    question = st.text_input(
        "💬 输入您的问题：",
        placeholder="例如：这份文档的核心内容是什么？有哪些关键知识点？",
        label_visibility="collapsed",
    )

    col_send, col_clear = st.columns([1, 5])
    with col_send:
        send_btn = st.button("🚀 提问", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if send_btn and question.strip():
        if not config.get("api_key"):
            st.error("请先配置 API Key！")
        else:
            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

            with st.spinner("🤔 正在检索知识库并生成回答..."):
                try:
                    result = st.session_state.rag_engine.ask(question)
                    answer = result["answer"]
                    sources = result.get("sources", [])

                    # 添加助手消息
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 问答失败：{str(e)}")
                    st.session_state.messages.pop()  # 移除用户消息


# ── 页脚 ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:13px;'>"
    "RAG 知识库问答系统 · Retrieval-Augmented Generation · "
    "Powered by 智库AI &nbsp;|&nbsp; "
    "<a href='https://github.com/ASJ-Alita/rag-knowledge-base' style='color:#64748b'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
