# -*- coding: utf-8 -*-
"""
文档解析模块 - 支持 PDF / TXT / DOCX
"""
import os
import re


def extract_text_from_file(filepath: str) -> str:
    """
    根据文件类型提取纯文本
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        return _extract_txt(filepath)
    elif ext == ".pdf":
        return _extract_pdf(filepath)
    elif ext in [".docx", ".doc"]:
        return _extract_docx(filepath)
    else:
        raise ValueError(f"不支持的文件格式：{ext}，目前支持 .txt / .pdf / .docx")


def _extract_txt(filepath: str) -> str:
    """从 TXT 文件提取文本"""
    encodings = ["utf-8", "gbk", "gb2312", "utf-16"]
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # 最后兜底：忽略编码错误
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _extract_pdf(filepath: str) -> str:
    """从 PDF 文件提取文本（使用 pdfplumber 或 PyPDF2）"""
    text = ""

    # 方法1: pdfplumber（推荐，保留布局）
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        if text.strip():
            return _clean_text(text)
    except ImportError:
        pass

    # 方法2: PyPDF2（备用）
    try:
        import PyPDF2
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        if text.strip():
            return _clean_text(text)
    except ImportError:
        pass

    raise ImportError(
        "PDF解析需要安装 pdfplumber 或 PyPDF2\n"
        "请运行：pip install pdfplumber\n"
        "或：pip install PyPDF2"
    )


def _extract_docx(filepath: str) -> str:
    """从 DOCX 文件提取文本"""
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        return _clean_text(text)
    except ImportError:
        raise ImportError(
            "DOCX解析需要安装 python-docx\n"
            "请运行：pip install python-docx"
        )


def _clean_text(text: str) -> str:
    """清理文本：去除多余空白、统一换行"""
    # 替换多个空格为单个空格
    text = re.sub(r"[ \t]+", " ", text)
    # 替换多个换行为单个换行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去除行首行尾空白
    lines = [line.strip() for line in text.split("\n")]
    # 过滤空行（保留最多连续2个空行）
    cleaned = []
    blank_count = 0
    for line in lines:
        if line:
            cleaned.append(line)
            blank_count = 0
        else:
            blank_count += 1
            if blank_count <= 1:
                cleaned.append("")
    return "\n".join(cleaned).strip()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list:
    """
    将长文本分块（滑动窗口）
    chunk_size: 每块字符数（含重叠）
    overlap: 块之间的重叠字符数
    """
    if len(text) <= chunk_size:
        return [{"text": text, "chunk_id": 0}]

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # 在句子边界处截断（更自然）
        if end < len(text):
            # 找最后一个句号/问号/感叹号
            for sep in ["。", "！", "？", ".\n", "?\n", "!\n", "\n"]:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size * 0.6:  # 确保不切太短
                    end = start + last_sep + len(sep)
                    chunk = text[start:end]
                    break

        chunks.append({
            "text": chunk.strip(),
            "chunk_id": chunk_id,
            "start_char": start,
            "end_char": end,
        })
        chunk_id += 1
        start = end - overlap  # 滑动窗口

    return chunks


def get_file_info(filepath: str) -> dict:
    """获取文件基本信息"""
    stat = os.stat(filepath)
    return {
        "name": os.path.basename(filepath),
        "size": stat.st_size,
        "size_kb": round(stat.st_size / 1024, 1),
        "type": os.path.splitext(filepath)[1].lower(),
        "modified": stat.st_mtime,
    }
