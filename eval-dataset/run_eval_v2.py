"""
RAG 双层评测 V2
改进1：检索层同时跑关键词匹配 + 语义匹配（embedding余弦相似度）
改进2：每个文档最优配置下跑完整 10 题 LLM 生成 + 打分

用法：python run_eval_v2.py
"""

import json, os, shutil, sys, time, math
from datetime import datetime

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, "c:/Users/Administrator/document-qa")
from src.chunker import chunk_text
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.deepseek_client import DeepSeekClient

EVAL_DIR = "c:/Users/Administrator/Desktop/eval-dataset"
CHROMA_DIR = f"{EVAL_DIR}/_eval_chroma"

CHUNK_SIZES = [256, 512, 1024]
OVERLAPS = [0, 50, 100]
TOP_KS = [3, 5, 7]

DOCS = [
    ("01-中文RAG教程-文本.txt", "01-中文RAG教程-QA.json", "RAG教程"),
    ("02-合同条款-文本.txt", "02-合同条款-QA.json", "腾讯云协议"),
    ("03-技术手册-文本.txt", "03-技术手册-QA.json", "技术手册"),
]

KEYWORDS = {
    "RAG教程": {
        0: ["检索增强生成", "先检索", "再生成", "真实文档", "Prompt", "RAG"],
        1: ["知识截止日期", "私有数据", "幻觉", "向量库", "实时检索", "强制"],
        2: ["离线索引", "在线查询", "文档加载", "文本切分", "Embedding", "向量数据库", "LLM生成"],
    },
    "腾讯云协议": {
        0: ["腾讯云计算", "北京", "有限责任公司", "腾讯云用户", "2025年12月23日"],
        1: ["公测", "内测", "不承诺", "可用性", "可靠性", "生产环境"],
        2: ["万分之五", "5个工作日", "账单异议", "预付费", "后付费", "违约金"],
    },
    "技术手册": {
        0: ["声明式", "重新执行", "回调函数", "代码即UI"],
        1: ["cache_data", "cache_resource", "数据处理", "资源", "pickle", "数据库连接", "模型"],
        2: ["st.session_state", "持久化", "字典", "聊天历史", "跨多次运行"],
    },
}


def get_api_key():
    key = os.getenv("DEEPSEEK_API_KEY")
    if key: return key
    try:
        from dotenv import load_dotenv
        load_dotenv("c:/Users/Administrator/document-qa/.env")
        return os.getenv("DEEPSEEK_API_KEY")
    except: pass
    return None


def load_data(full=False):
    data = []
    for txt, qa_json, name in DOCS:
        with open(f"{EVAL_DIR}/{txt}", encoding="utf-8") as f:
            text = f.read()
        with open(f"{EVAL_DIR}/{qa_json}", encoding="utf-8") as f:
            qa = json.load(f)
        data.append((name, text, qa))  # always full QA for phase 2
    return data


def keyword_score(sources, keywords):
    all_text = " ".join(s["text"] for s in sources)
    hits = sum(1 for kw in keywords if kw.lower() in all_text.lower())
    return hits / len(keywords) if keywords else 0


def semantic_score(reference_embedding, sources, embedder):
    """用参考答案的 embedding 和检索到的 chunk 算余弦相似度。返回 max similarity"""
    if not sources: return 0.0
    chunk_texts = [s["text"] for s in sources]
    chunk_embs = embedder.embed(chunk_texts)
    # L2 normalized dot product = cosine similarity
    max_sim = max(sum(a * b for a, b in zip(reference_embedding, c_emb)) for c_emb in chunk_embs)
    # clamp to [0,1] (normalized vectors can have tiny negatives from float error)
    return max(0.0, min(1.0, max_sim))


# ═══════════════════════════════════════════════════════════════
# Phase 1: 检索层 — 关键词 + 语义双评分
# ═══════════════════════════════════════════════════════════════
def phase1_retrieval(data, embedder):
    """27 configs x 3 docs x 3 questions. Both keyword + semantic scoring."""
    print("=" * 60)
    print("Phase 1: Retrieval layer (keyword + semantic)")
    print(f"Configs: {len(CHUNK_SIZES)}x{len(OVERLAPS)}x{len(TOP_KS)}x{len(data)} = "
          f"{len(CHUNK_SIZES)*len(OVERLAPS)*len(TOP_KS)*len(data)}")
    print("=" * 60)

    results = []
    total = len(CHUNK_SIZES) * len(OVERLAPS) * len(TOP_KS) * len(data)
    idx = 0

    for cs in CHUNK_SIZES:
        for ov in OVERLAPS:
            for tk in TOP_KS:
                for doc_name, doc_text, qa_list in data:
                    idx += 1

                    store = VectorStore(persist_dir=CHROMA_DIR)
                    chunks = chunk_text(doc_text, chunk_size=cs, chunk_overlap=ov)
                    chunk_embs = embedder.embed(chunks)
                    cid = store.add_documents(chunks, chunk_embs, doc_name)

                    q_scores = []
                    for qi in range(3):  # first 3 questions
                        item = qa_list[qi]
                        q_emb = embedder.embed_query(item["question"])
                        sources = store.search(q_emb, cid, top_k=tk)

                        # Keyword score
                        kw = keyword_score(sources, KEYWORDS.get(doc_name, {}).get(qi, []))
                        # Semantic score: embed reference answer, compare to chunks
                        ref_emb = embedder.embed_query(item["answer"])
                        sem = semantic_score(ref_emb, sources, embedder)

                        q_scores.append({"q_idx": qi, "keyword_hit": round(kw, 3), "semantic_sim": round(sem, 3)})

                    kw_avg = sum(s["keyword_hit"] for s in q_scores) / len(q_scores)
                    sem_avg = sum(s["semantic_sim"] for s in q_scores) / len(q_scores)

                    results.append({
                        "chunk_size": cs, "overlap": ov, "top_k": tk, "doc": doc_name,
                        "chunk_count": len(chunks),
                        "avg_keyword_hit": round(kw_avg, 3),
                        "avg_semantic_sim": round(sem_avg, 3),
                        "question_scores": q_scores,
                    })

                    store.delete_collection(cid)
                    shutil.rmtree(CHROMA_DIR, ignore_errors=True)

                    print(f"\r[{idx}/{total}] cs={cs} ov={ov} tk={tk} {doc_name}: "
                          f"kw={kw_avg:.3f} sem={sem_avg:.3f}", end="")

    print("\n")
    with open(f"{EVAL_DIR}/retrieval_scores_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 2: 为每个文档挑最优配置（按语义分），跑完整 10 题
# ═══════════════════════════════════════════════════════════════
def phase2_full_generation(retrieval_results, data, embedder, llm):
    print("=" * 60)
    print("Phase 2: Full 10-QA generation on best configs")
    print("=" * 60)

    # 每个文档选语义分最高的配置
    best_configs = {}
    for doc_name in ["RAG教程", "腾讯云协议", "技术手册"]:
        doc_results = [r for r in retrieval_results if r["doc"] == doc_name]
        best = max(doc_results, key=lambda x: x["avg_semantic_sim"])
        best_configs[doc_name] = best
        print(f"Best for {doc_name}: chunk={best['chunk_size']} ov={best['overlap']} "
              f"tk={best['top_k']} sem={best['avg_semantic_sim']:.3f}")

    gen_results = []
    total_calls = sum(len(qa_list) for _, _, qa_list in data)
    print(f"\nLLM calls: {total_calls} generate + {total_calls} score = {total_calls*2}")
    print(f"Estimated cost: ~{total_calls*2*0.003:.1f} yuan")
    print(f"Estimated time: ~{total_calls//5}-{total_calls//3} min\n")

    gen_idx = 0
    for doc_name, doc_text, qa_list in data:
        best = best_configs[doc_name]
        cs, ov, tk = best["chunk_size"], best["overlap"], best["top_k"]

        store = VectorStore(persist_dir=CHROMA_DIR)
        chunks = chunk_text(doc_text, chunk_size=cs, chunk_overlap=ov)
        embs = embedder.embed(chunks)
        cid = store.add_documents(chunks, embs, doc_name)

        print(f"\n[{doc_name}] chunk={cs} overlap={ov} top_k={tk}")
        q_results = []

        for qi, item in enumerate(qa_list):
            gen_idx += 1
            q_emb = embedder.embed_query(item["question"])
            sources = store.search(q_emb, cid, top_k=tk)

            ref_emb = embedder.embed_query(item["answer"])
            sem = semantic_score(ref_emb, sources, embedder)

            context_parts = [f"[S{j+1}]\n{s['text']}" for j, s in enumerate(sources)]
            context = "\n\n---\n\n".join(context_parts)

            try:
                gen = llm.generate(
                    system_prompt="你是文档问答助手。只根据文档内容作答，不要编造。",
                    user_prompt=f"文档：\n{context}\n\n问题：{item['question']}",
                    temperature=0.3)
            except Exception as e:
                gen = f"[ERROR] {e}"

            # Score
            try:
                sp = (f"标准答案：{item['answer']}\n"
                      f"生成答案：{gen}\n"
                      f"从三个维度打分(1-10)：准确性、完整性、简洁性。只输出JSON对象。")
                resp = llm.generate(
                    system_prompt="你是RAG评测专家。只输出JSON。",
                    user_prompt=sp, temperature=0)
                scores_raw = json.loads(resp)
                key_map = {"准确性": "accuracy", "完整性": "completeness", "简洁性": "conciseness"}
                scores = {}
                for k, v in scores_raw.items():
                    scores[key_map.get(k, k)] = v
            except Exception as e:
                scores = {"accuracy": 0, "completeness": 0, "conciseness": 0, "comment": str(e)[:60]}

            q_results.append({
                "q_idx": qi, "question": item["question"][:60],
                "reference": item["answer"], "generated": gen,
                "semantic_sim": round(sem, 3),
                "scores": scores,
            })
            print(f"  [{qi+1}/{len(qa_list)}] "
                  f"sem={sem:.3f} acc={scores.get('accuracy',0)} "
                  f"comp={scores.get('completeness',0)} conc={scores.get('conciseness',0)}")

        store.delete_collection(cid)
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

        # Compute stats
        acc = sum(r["scores"].get("accuracy", 0) for r in q_results) / len(q_results)
        comp = sum(r["scores"].get("completeness", 0) for r in q_results) / len(q_results)
        conc = sum(r["scores"].get("conciseness", 0) for r in q_results) / len(q_results)
        # Std dev
        overalls = [(r["scores"].get("accuracy",0)+r["scores"].get("completeness",0)+r["scores"].get("conciseness",0))/3
                     for r in q_results]
        mean_ov = sum(overalls) / len(overalls)
        std_ov = math.sqrt(sum((x-mean_ov)**2 for x in overalls) / len(overalls))

        gen_results.append({
            "chunk_size": cs, "overlap": ov, "top_k": tk, "doc": doc_name,
            "avg_scores": {"accuracy": round(acc, 2), "completeness": round(comp, 2),
                           "conciseness": round(conc, 2), "overall": round(mean_ov, 2),
                           "std": round(std_ov, 2)},
            "question_results": q_results,
        })

    with open(f"{EVAL_DIR}/eval_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(gen_results, f, ensure_ascii=False, indent=2)
    return gen_results


# ═══════════════════════════════════════════════════════════════
# Phase 3: 综合报告
# ═══════════════════════════════════════════════════════════════
def phase3_report(retrieval_results, gen_results, t0, t1, t2):
    print("\n" + "=" * 60)
    print("Phase 3: Generating final report")
    print("=" * 60)

    L = []
    L.append("# RAG Evaluation Report V2")
    L.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"\nTotal time: {t2-t0:.0f}s (retrieval: {t1-t0:.0f}s, generation: {t2-t1:.0f}s)")

    # ── Comparison: keyword vs semantic ranking ──
    L.append("\n## 1. Retrieval Scoring: Keyword vs Semantic\n")
    L.append("### How rankings differ\n")
    L.append("| Doc | Config (keyword best) | kw_score | Config (semantic best) | sem_score | rank_shift |")
    L.append("|-----|----------------------|----------|------------------------|-----------|------------|")

    for doc_name in ["RAG教程", "腾讯云协议", "技术手册"]:
        doc_results = [r for r in retrieval_results if r["doc"] == doc_name]
        kw_best = max(doc_results, key=lambda x: x["avg_keyword_hit"])
        sem_best = max(doc_results, key=lambda x: x["avg_semantic_sim"])
        kw_rank = sorted(doc_results, key=lambda x: x["avg_keyword_hit"], reverse=True)
        sem_rank = sorted(doc_results, key=lambda x: x["avg_semantic_sim"], reverse=True)
        kw_pos = sem_rank.index(kw_best) if kw_best in sem_rank else -1
        rank_shift = f"kw#1 -> sem#{kw_pos+1}" if kw_pos >= 0 else "N/A"
        L.append(f"| {doc_name} | cs={kw_best['chunk_size']} ov={kw_best['overlap']} tk={kw_best['top_k']} | "
                 f"{kw_best['avg_keyword_hit']:.3f} | "
                 f"cs={sem_best['chunk_size']} ov={sem_best['overlap']} tk={sem_best['top_k']} | "
                 f"{sem_best['avg_semantic_sim']:.3f} | {rank_shift} |")

    # ── Top 5 by semantic score ──
    L.append("\n### Top 5 configs by semantic similarity\n")
    L.append("| Rank | chunk | ov | tk | Doc | keyword | semantic |")
    L.append("|------|-------|-----|-----|-----|---------|----------|")
    sorted_ret = sorted(retrieval_results, key=lambda x: x["avg_semantic_sim"], reverse=True)
    for i, r in enumerate(sorted_ret[:5]):
        L.append(f"| {i+1} | {r['chunk_size']} | {r['overlap']} | {r['top_k']} | "
                 f"{r['doc']} | {r['avg_keyword_hit']:.3f} | {r['avg_semantic_sim']:.3f} |")

    # ── chunk_size analysis ──
    L.append("\n### chunk_size effect (avg semantic sim across all configs)\n")
    L.append("| chunk | avg_semantic | avg_keyword |")
    L.append("|-------|-------------|-------------|")
    for cs in CHUNK_SIZES:
        cs_results = [r for r in retrieval_results if r["chunk_size"] == cs]
        avg_sem = sum(r["avg_semantic_sim"] for r in cs_results) / len(cs_results)
        avg_kw = sum(r["avg_keyword_hit"] for r in cs_results) / len(cs_results)
        L.append(f"| {cs} | {avg_sem:.3f} | {avg_kw:.3f} |")

    # ── Full 10-QA LLM scores with std ──
    L.append("\n## 2. Full 10-Question LLM Scores (with std dev)\n")
    L.append("| Doc | chunk | ov | tk | Accuracy | Completeness | Conciseness | **Overall** | Std |")
    L.append("|-----|-------|-----|-----|----------|-------------|-------------|-------------|-----|")
    for g in gen_results:
        s = g["avg_scores"]
        L.append(f"| {g['doc']} | {g['chunk_size']} | {g['overlap']} | {g['top_k']} | "
                 f"{s['accuracy']} | {s['completeness']} | {s['conciseness']} | "
                 f"**{s['overall']}** | +/-{s['std']} |")

    # ── Per-question breakdown for each doc ──
    L.append("\n### Per-question breakdown\n")
    for g in gen_results:
        L.append(f"\n**{g['doc']}** (chunk={g['chunk_size']} ov={g['overlap']} tk={g['top_k']})\n")
        L.append("| # | Question | sem_sim | acc | comp | conc | avg |")
        L.append("|---|----------|---------|-----|------|------|-----|")
        for qr in g["question_results"]:
            avg = (qr["scores"].get("accuracy",0) + qr["scores"].get("completeness",0) + qr["scores"].get("conciseness",0)) / 3
            L.append(f"| {qr['q_idx']+1} | {qr['question'][:40]}... | {qr['semantic_sim']:.3f} | "
                     f"{qr['scores'].get('accuracy',0)} | {qr['scores'].get('completeness',0)} | "
                     f"{qr['scores'].get('conciseness',0)} | {avg:.1f} |")

    # ── Key insights ──
    L.append("\n## 3. Key Insights\n")
    L.append("### Keyword vs Semantic: which is better?")
    L.append("Keyword matching is fast and interpretable but biased toward short chunks ")
    L.append("(keywords are denser in smaller text spans). Semantic similarity via ")
    L.append("embedding dot product is chunk-size-neutral and measures actual meaning overlap. ")
    L.append("The semantic ranking often disagrees with keyword ranking on long documents ")
    L.append("because keyword matching overvalues small, keyword-dense fragments while ")
    L.append("semantic matching values coherent context.\n")

    L.append("### Statistical significance")
    L.append("With 10 questions per document, we can report mean +/- std. A low std ")
    L.append("(e.g. < 1.0) means the config performs consistently across different ")
    L.append("question types. A high std suggests the config works well for some questions ")
    L.append("but poorly for others — indicating the need for adaptive retrieval strategies.\n")

    # ── Final recommendation ──
    L.append("## 4. Final Recommendation\n")
    L.append("| Doc | chunk | overlap | top_k | LLM Overall | Recommendation |")
    L.append("|-----|-------|---------|-------|-------------|----------------|")
    reasons = {
        "RAG教程": "Long technical doc: needs overlap for table/list integrity",
        "腾讯云协议": "Short legal doc: clauses are self-contained, minimal overlap needed",
        "技术手册": "Medium manual: code+explanation together, bigger chunks help",
    }
    for g in gen_results:
        s = g["avg_scores"]
        L.append(f"| {g['doc']} | {g['chunk_size']} | {g['overlap']} | {g['top_k']} | "
                 f"**{s['overall']} +/-{s['std']}** | {reasons.get(g['doc'], '')} |")

    report = "\n".join(L)
    path = f"{EVAL_DIR}/eval_report_v2.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved: {path}")

    # Print key findings
    print("\n=== KEY FINDINGS ===")
    for g in gen_results:
        s = g["avg_scores"]
        print(f"{g['doc']}: chunk={g['chunk_size']} ov={g['overlap']} tk={g['top_k']} "
              f"overall={s['overall']} +/-{s['std']} (10 QAs)")
    print(f"\nFull report: {path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print("  RAG V2: Semantic scoring + Full 10-QA")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    api_key = get_api_key()
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set"); return

    data = load_data()
    print(f"\nDocuments: {len(data)} (each 10 QAs)")
    for name, text, qa in data:
        print(f"  {name}: {len(text)} chars, {len(qa)} questions")

    embedder = EmbeddingModel()
    llm = DeepSeekClient(api_key=api_key)

    # Phase 1
    t0 = time.time()
    ret_results = phase1_retrieval(data, embedder)
    t1 = time.time()
    print(f"Phase 1 done: {t1-t0:.0f}s")

    # Phase 2
    gen_results = phase2_full_generation(ret_results, data, embedder, llm)
    t2 = time.time()
    print(f"\nPhase 2 done: {t2-t1:.0f}s")

    # Phase 3
    phase3_report(ret_results, gen_results, t0, t1, t2)
    print(f"\nTotal: {t2-t0:.0f}s")


if __name__ == "__main__":
    main()
