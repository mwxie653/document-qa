# RAG Evaluation Report V2

Generated: 2026-05-12 12:06:19

Total time: 203s (retrieval: 112s, generation: 90s)

## 1. Retrieval Scoring: Keyword vs Semantic

### How rankings differ

| Doc | Config (keyword best) | kw_score | Config (semantic best) | sem_score | rank_shift |
|-----|----------------------|----------|------------------------|-----------|------------|
| RAG教程 | cs=256 ov=100 tk=3 | 0.905 | cs=256 ov=100 tk=3 | 0.723 | kw#1 -> sem#1 |
| 腾讯云协议 | cs=256 ov=0 tk=5 | 0.944 | cs=256 ov=0 tk=5 | 0.634 | kw#1 -> sem#1 |
| 技术手册 | cs=512 ov=0 tk=7 | 0.933 | cs=512 ov=0 tk=7 | 0.804 | kw#1 -> sem#1 |

### Top 5 configs by semantic similarity

| Rank | chunk | ov | tk | Doc | keyword | semantic |
|------|-------|-----|-----|-----|---------|----------|
| 1 | 512 | 0 | 7 | 技术手册 | 0.933 | 0.804 |
| 2 | 1024 | 0 | 5 | 技术手册 | 0.933 | 0.804 |
| 3 | 1024 | 0 | 7 | 技术手册 | 0.933 | 0.804 |
| 4 | 512 | 50 | 7 | 技术手册 | 0.933 | 0.781 |
| 5 | 256 | 0 | 3 | 技术手册 | 0.583 | 0.780 |

### chunk_size effect (avg semantic sim across all configs)

| chunk | avg_semantic | avg_keyword |
|-------|-------------|-------------|
| 256 | 0.691 | 0.748 |
| 512 | 0.643 | 0.735 |
| 1024 | 0.619 | 0.861 |

## 2. Full 10-Question LLM Scores (with std dev)

| Doc | chunk | ov | tk | Accuracy | Completeness | Conciseness | **Overall** | Std |
|-----|-------|-----|-----|----------|-------------|-------------|-------------|-----|
| RAG教程 | 256 | 100 | 3 | 7.0 | 6.5 | 8.6 | **7.37** | +/-2.26 |
| 腾讯云协议 | 256 | 0 | 5 | 9.4 | 8.5 | 8.1 | **8.67** | +/-0.7 |
| 技术手册 | 512 | 0 | 7 | 8.7 | 8.2 | 8.8 | **8.57** | +/-1.65 |

### Per-question breakdown


**RAG教程** (chunk=256 ov=100 tk=3)

| # | Question | sem_sim | acc | comp | conc | avg |
|---|----------|---------|-----|------|------|-----|
| 1 | RAG是什么？它的核心思路是什么？... | 0.883 | 10 | 10 | 9 | 9.7 |
| 2 | LLM有哪三大局限？RAG分别如何解决？... | 0.706 | 10 | 10 | 9 | 9.7 |
| 3 | RAG的完整工作流程分为哪两个阶段？每个阶段包含哪些步骤？... | 0.580 | 8 | 6 | 9 | 7.7 |
| 4 | 文本切分有哪四种常见策略？... | 0.644 | 7 | 6 | 8 | 7.0 |
| 5 | RAG在生产环境部署有哪些最佳实践？... | 0.782 | 7 | 6 | 8 | 7.0 |
| 6 | 什么是MMR？它在RAG中起什么作用？... | 0.439 | 1 | 1 | 10 | 4.0 |
| 7 | RAG系统中Prompt设计有哪些原则？... | 0.433 | 10 | 10 | 9 | 9.7 |
| 8 | 多轮对话RAG中，如何解决用户提问指代不明的问题？... | 0.707 | 10 | 10 | 9 | 9.7 |
| 9 | RAG检索模块有哪些优化技巧？... | 0.516 | 1 | 1 | 8 | 3.3 |
| 10 | RAG和LLM长上下文（如128K token）有什么区别？什么时候用RAG更好... | 0.546 | 6 | 5 | 7 | 6.0 |

**腾讯云协议** (chunk=256 ov=0 tk=5)

| # | Question | sem_sim | acc | comp | conc | avg |
|---|----------|---------|-----|------|------|-----|
| 1 | 本协议的签约主体是谁？最新版本什么时候生效？... | 0.873 | 10 | 10 | 9 | 9.7 |
| 2 | 腾讯云对试用服务（公测或内测）有什么特别说明？... | 0.406 | 10 | 10 | 8 | 9.3 |
| 3 | 用户欠费后怎么处理？违约金利率是多少？... | 0.623 | 10 | 6 | 9 | 8.3 |
| 4 | 腾讯云在什么情况下可以中止或终止服务？重大调整需要提前多久通知？... | 0.879 | 8 | 6 | 9 | 7.7 |
| 5 | 用户存储在腾讯云上的数据归谁所有？服务终止后数据怎么处理？... | 0.761 | 9 | 8 | 9 | 8.7 |
| 6 | 第五条网络安全和网络秩序中，禁止了哪些内容？... | 0.685 | 9 | 8 | 6 | 7.7 |
| 7 | 腾讯云的赔偿上限是多少？哪些损失不承担？... | 0.768 | 9 | 9 | 8 | 8.7 |
| 8 | 争议解决适用什么法律？协商不成去哪个法院？... | 0.791 | 10 | 10 | 8 | 9.3 |
| 9 | 保密义务持续多久？什么信息不算保密信息？... | 0.872 | 10 | 10 | 8 | 9.3 |
| 10 | 用户授权他人管理账号需要注意什么？... | 0.623 | 9 | 8 | 7 | 8.0 |

**技术手册** (chunk=512 ov=0 tk=7)

| # | Question | sem_sim | acc | comp | conc | avg |
|---|----------|---------|-----|------|------|-----|
| 1 | Streamlit的数据流模型是什么？用户交互时发生了什么？... | 0.700 | 10 | 10 | 9 | 9.7 |
| 2 | @st.cache_data和@st.cache_resource有什么区别？分... | 0.844 | 10 | 10 | 9 | 9.7 |
| 3 | 如何在Streamlit中持久化用户数据跨多次交互？... | 0.869 | 10 | 9 | 8 | 9.0 |
| 4 | Streamlit和FastAPI如何配合使用？典型架构是什么？... | 0.670 | 9 | 8 | 9 | 8.7 |
| 5 | 如何在Streamlit中实现多页面应用？... | 0.864 | 10 | 9 | 9 | 9.3 |
| 6 | Streamlit怎么实现聊天界面？用到哪些组件？... | 0.618 | 9 | 8 | 7 | 8.0 |
| 7 | Docker部署Streamlit时需要注意什么？... | 0.424 | 1 | 1 | 10 | 4.0 |
| 8 | Streamlit如何做性能优化？有哪些缓存策略？... | 0.942 | 10 | 10 | 9 | 9.7 |
| 9 | 如何自定义Streamlit的主题和外观？... | 0.742 | 10 | 10 | 9 | 9.7 |
| 10 | 文件上传组件st.file_uploader怎么用？支持哪些文件类型？... | 0.578 | 8 | 7 | 9 | 8.0 |

## 3. Key Insights

### Keyword vs Semantic: which is better?
Keyword matching is fast and interpretable but biased toward short chunks 
(keywords are denser in smaller text spans). Semantic similarity via 
embedding dot product is chunk-size-neutral and measures actual meaning overlap. 
The semantic ranking often disagrees with keyword ranking on long documents 
because keyword matching overvalues small, keyword-dense fragments while 
semantic matching values coherent context.

### Statistical significance
With 10 questions per document, we can report mean +/- std. A low std 
(e.g. < 1.0) means the config performs consistently across different 
question types. A high std suggests the config works well for some questions 
but poorly for others — indicating the need for adaptive retrieval strategies.

## 4. Final Recommendation

| Doc | chunk | overlap | top_k | LLM Overall | Recommendation |
|-----|-------|---------|-------|-------------|----------------|
| RAG教程 | 256 | 100 | 3 | **7.37 +/-2.26** | Long technical doc: needs overlap for table/list integrity |
| 腾讯云协议 | 256 | 0 | 5 | **8.67 +/-0.7** | Short legal doc: clauses are self-contained, minimal overlap needed |
| 技术手册 | 512 | 0 | 7 | **8.57 +/-1.65** | Medium manual: code+explanation together, bigger chunks help |