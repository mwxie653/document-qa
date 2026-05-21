"""Streamlit frontend for the Document Q&A system."""

import os
import streamlit as st
from dotenv import load_dotenv
from src.rag_engine import RAGEngine

load_dotenv()

st.set_page_config(page_title="智能文档问答", page_icon="", layout="wide")

# ── 初始化 session state ──────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "collection_id" not in st.session_state:
    st.session_state.collection_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── 侧边栏 ────────────────────────────────────────────────────────
with st.sidebar:
    st.title(" 智能文档问答")

    # API Key
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        help="从 platform.deepseek.com/api_keys 获取",
    )

    if not api_key:
        st.warning("请输入 API Key")
        st.stop()

    # Reranker 开关
    st.divider()
    st.subheader(" 高级设置")
    use_reranker = st.toggle(
        "启用重排序 (Reranker)",
        value=st.session_state.get("use_reranker", False),
        help="用 Cross-Encoder 对检索结果精排，提升精度。首次启用需下载模型（约1GB）。",
    )
    st.caption("粗检索取 top_k×3，经 reranker 精排后保留 top_k")
    if use_reranker != st.session_state.get("_prev_reranker", None):
        st.session_state.rag = None  # 强制重建引擎
        st.session_state._prev_reranker = use_reranker
        st.session_state.use_reranker = use_reranker

    # 初始化 RAG 引擎
    if st.session_state.rag is None:
        st.session_state.rag = RAGEngine(
            deepseek_api_key=api_key,
            use_reranker=use_reranker,
        )

    st.divider()

    # 文件上传
    st.subheader(" 上传文档")
    uploaded_file = st.file_uploader(
        "支持 PDF / Word",
        type=["pdf", "docx", "doc"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        with st.spinner("正在解析并索引文档..."):
            try:
                file_bytes = uploaded_file.read()
                collection_id = st.session_state.rag.index_document(
                    file_bytes, uploaded_file.name
                )
                st.session_state.collection_id = collection_id
                st.session_state.chat_history = []
                st.success(f"已索引：{uploaded_file.name}")
                st.info(f"分块 ID: `{collection_id}`")
            except Exception as e:
                st.error(f"解析失败：{e}")

    st.divider()

    # 已索引文档列表
    st.subheader(" 已索引文档")
    try:
        docs = st.session_state.rag.list_documents() if st.session_state.rag else []
        if docs:
            for doc in docs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f" {doc['count']} 块")
                with col2:
                    if st.button("删除", key=f"del-{doc['name']}"):
                        st.session_state.rag.delete_document(doc["name"])
                        if st.session_state.collection_id == doc["name"]:
                            st.session_state.collection_id = None
                            st.session_state.chat_history = []
                        st.rerun()
                st.caption(f"`{doc['name']}`")
        else:
            st.caption("暂无已索引文档")
    except Exception:
        st.caption("暂无已索引文档")

# ── 主区域 ────────────────────────────────────────────────────────
st.title("智能文档问答系统")
st.caption("上传 PDF 或 Word 文档，然后用自然语言提问 —— 答案会附带原文引用。")

# 检查是否已索引文档
if st.session_state.collection_id is None:
    st.info(" 请在左侧上传一份文档开始使用。")
    st.stop()

# 聊天历史
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(msg["question"])
    with st.chat_message("assistant"):
        st.markdown(msg["answer"])
        if msg.get("sources"):
            with st.expander(" 查看原文引用"):
                for i, s in enumerate(msg["sources"]):
                    st.caption(f"片段 {i+1}（相似度: {1 - s['distance']:.3f}）")
                    st.text(s["text"])
                    st.divider()

# 输入问题
if question := st.chat_input("输入你的问题..."):
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                result = st.session_state.rag.query(
                    question,
                    st.session_state.collection_id,
                    top_k=5,
                )
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander(" 查看原文引用"):
                        for i, s in enumerate(result["sources"]):
                            st.caption(f"片段 {i+1}（相似度: {1 - s['distance']:.3f}）")
                            st.text(s["text"])
                            st.divider()
            except Exception as e:
                st.error(f"查询失败：{e}")
                result = {"answer": f"查询失败：{e}", "sources": []}

    st.session_state.chat_history.append({
        "question": question,
        "answer": result["answer"],
        "sources": result.get("sources", []),
    })
