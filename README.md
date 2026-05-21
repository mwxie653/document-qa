# 智能文档问答系统 (Document Q&A with RAG)

基于 RAG（Retrieval-Augmented Generation）架构的智能文档问答系统。上传 PDF/Word 文档，用自然语言提问，系统自动检索相关段落并生成带原文引用的答案。

## 技术栈

| 环节 | 选型 | 说明 |
|------|------|------|
| 文档解析 | PyMuPDF + python-docx | PDF 逐页提取 / Word 逐段提取 |
| 文本分块 | LangChain RecursiveCharacterTextSplitter | 中英文分隔符，递归语义切分 |
| 向量化 | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) | 384维，中英双语，本地运行 |
| 向量存储 | ChromaDB | 持久化本地存储，无需外部服务 |
| 重排序 | BAAI/bge-reranker-base | Cross-Encoder 精排，可选开启 |
| 大模型 | DeepSeek API (OpenAI 兼容) | deepseek-chat，性价比高 |
| 前端 | Streamlit | 对话式 UI，文件上传 + 实时问答 |
| 部署 | Docker + docker-compose | 一键启动 |

## 快速开始

### 1. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key（从 https://platform.deepseek.com/api_keys 获取）
# 国内用户设 HF_MIRROR=1，使用 HuggingFace 镜像下载模型
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动

```bash
streamlit run app.py
# 浏览器打开 http://localhost:8501
```

### 4. Docker 部署

```bash
docker-compose up --build
```

## 项目结构

```
document-qa/
├── app.py                    # Streamlit 前端
├── src/
│   ├── document_parser.py    # PDF/Word → 纯文本
│   ├── chunker.py            # 文本 → 分块
│   ├── embeddings.py         # 分块 → 向量
│   ├── vector_store.py       # Chroma 增删查
│   ├── reranker.py           # Cross-Encoder 重排序
│   ├── deepseek_client.py    # DeepSeek API 封装
│   └── rag_engine.py         # RAG 编排引擎
├── eval-dataset/             # 评测数据集与脚本
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── .gitignore
```

## RAG Pipeline

```
离线索引阶段：
  文档上传 → PDF/Word解析 → 递归分块 → Embedding向量化 → 存入Chroma

在线查询阶段：
  用户提问 → 问题向量化 → Chroma检索(×3候选) → [Reranker精排] → 拼接Prompt → DeepSeek → 答案+引用
```

## 评测结果

### 评测设计

- **双层架构**：检索层（零API成本，81个配置点）→ 生成层（仅最优配置跑LLM）
- **双指标**：关键词命中率 + 语义相似度（embedding余弦相似度）——修正单一指标的偏差
- **27种配置**：3 chunk_size × 3 overlap × 3 top_k，在3种不同类型文档上测试
- **统计显著性**：每份文档每配置10道题，报告均值和标准差

### 最优配置

| 文档类型 | chunk_size | overlap | top_k | LLM综合分 | 建议 |
|---------|-----------|---------|-------|----------|------|
| 长技术教程（20K字） | 256 | 100 | 5 | 7.37 ±2.26 | 小块+大重叠防表格截断 |
| 短法律合同（3K字） | 256 | 0 | 5 | 8.67 ±0.70 | 条款自包含，最稳定 |
| 中长技术手册（5K字） | 512 | 0 | 7 | 8.57 ±1.65 | 大块提供代码+说明完整上下文 |

### Reranker 效果

| 文档 | OFF | ON | 提升 |
|------|-----|-----|------|
| RAG教程（长文档） | 4.5 | 6.8 | **+51%** |
| 技术手册（中文档） | 7.7 | 9.7 | **+26%** |
| 腾讯云协议（短文档） | 8.3 | 8.6 | +4% |

> **结论**：长文档建议开启 Reranker（5s延迟换26-51%质量提升），短文档可关闭。

### 核心发现

1. **关键词 vs 语义评分存在系统性偏差**：关键词偏向大chunk（+39%虚高），仅用关键词会错选 chunk=1024 为最优
2. **文档结构 > 参数微调**：短文档 ±0.7 vs 长文档 ±2.26，先分析文档结构再调参
3. **语义相似度可预判生成质量**：sem_sim < 0.5 时 LLM 必崩，可用于线上 fail-fast
4. **预测准确率仅 19%**：基于领域知识的假设与实验数据存在显著差异——先假设再实验的科学评测流程本身比结论更值钱

## License

MIT
