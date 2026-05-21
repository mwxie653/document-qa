"""
Reranker 对比评测：重点测试之前失败的低分案例
在最优配置下，对比 reranker ON vs OFF 的效果
"""
import json, os, shutil, sys, time
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
CHROMA_DIR = f"{EVAL_DIR}/_eval_rerank"


def get_api_key():
    from dotenv import load_dotenv
    load_dotenv("c:/Users/Administrator/document-qa/.env")
    return os.getenv("DEEPSEEK_API_KEY")


def semantic_score(ref_emb, sources):
    """Cosine similarity between reference embedding and retrieved chunks"""
    if not sources: return 0.0
    chunk_texts = [s["text"] for s in sources]
    from src.embeddings import EmbeddingModel
    emb = EmbeddingModel()
    chunk_embs = emb.embed(chunk_texts)
    return max(sum(a*b for a,b in zip(ref_emb, e)) for e in chunk_embs)


def run_test(doc_name, doc_text, qa_list, config, embedder, llm, use_reranker):
    """Run QAs with or without reranker"""
    cs, ov, tk = config

    store = VectorStore(persist_dir=CHROMA_DIR)
    chunks = chunk_text(doc_text, chunk_size=cs, chunk_overlap=ov)
    embs = embedder.embed(chunks)
    cid = store.add_documents(chunks, embs, doc_name)

    results = []

    for qi, item in enumerate(qa_list):
        q_emb = embedder.embed_query(item["question"])

        if use_reranker:
            # 粗检索 3x，再重排
            from src.reranker import Reranker
            reranker = Reranker()
            sources = store.search(q_emb, cid, top_k=tk * 3)
            chunk_texts = [s["text"] for s in sources]
            reranked = reranker.rerank(item["question"], chunk_texts, top_k=tk)
            # Match back
            text_map = {s["text"]: s for s in sources}
            sources = []
            for r in reranked:
                s = text_map.get(r["text"], {"text": r["text"], "distance": 0})
                s["rerank_score"] = r["score"]
                sources.append(s)
        else:
            sources = store.search(q_emb, cid, top_k=tk)

        ref_emb = embedder.embed_query(item["answer"])
        sem = semantic_score(ref_emb, sources)

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
            sp = (f"标准答案：{item['answer']}\n生成答案：{gen}\n"
                  f"从三个维度打分(1-10)：准确性、完整性、简洁性。只输出JSON。")
            resp = llm.generate(system_prompt="你是RAG评测专家。只输出JSON。", user_prompt=sp, temperature=0)
            scores_raw = json.loads(resp)
            key_map = {"准确性": "accuracy", "完整性": "completeness", "简洁性": "conciseness"}
            scores = {key_map.get(k, k): v for k, v in scores_raw.items()}
        except Exception as e:
            scores = {"accuracy": 0, "completeness": 0, "conciseness": 0, "comment": str(e)[:60]}

        avg = (scores.get("accuracy",0) + scores.get("completeness",0) + scores.get("conciseness",0)) / 3
        results.append({
            "q_idx": qi, "question": item["question"][:60],
            "semantic_sim": round(sem, 3),
            "scores": scores, "avg": round(avg, 1),
        })

    store.delete_collection(cid)
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    return results


def main():
    api_key = get_api_key()
    if not api_key:
        print("ERROR: no API key"); return

    embedder = EmbeddingModel()
    llm = DeepSeekClient(api_key=api_key)

    # Test scenarios: (doc_name, config, focus_questions)
    # These are the worst-performing questions from V2
    scenarios = [
        ("RAG教程", "01-中文RAG教程-文本.txt", "01-中文RAG教程-QA.json",
         (256, 100, 3), [5, 8, 3]),  # Q6 MMR, Q9 retrieval, Q4 CityU
        ("技术手册", "03-技术手册-文本.txt", "03-技术手册-QA.json",
         (512, 0, 7), [6, 2, 4]),     # Q7 Docker, Q3 Session, Q5 multi-page
        ("腾讯云协议", "02-合同条款-文本.txt", "02-合同条款-QA.json",
         (256, 0, 5), [1, 3, 5]),     # Q2 trial, Q4 terminate, Q6 security
    ]

    print("=" * 60)
    print("Reranker A/B Test: OFF vs ON")
    print("=" * 60)

    all_results = []

    for doc_name, txt_file, qa_file, config, q_indices in scenarios:
        with open(f"{EVAL_DIR}/{txt_file}", encoding="utf-8") as f:
            doc_text = f.read()
        with open(f"{EVAL_DIR}/{qa_file}", encoding="utf-8") as f:
            qa_list = json.load(f)

        target_qs = [qa_list[i] for i in q_indices]
        cs, ov, tk = config

        print(f"\n{'='*60}")
        print(f"{doc_name} | config={cs}/{ov}/{tk} | {len(target_qs)} questions")
        print(f"{'='*60}")

        for mode, label in [(False, "OFF"), (True, "ON")]:
            print(f"\n--- Reranker {label} ---")
            t0 = time.time()

            # Need to pass use_reranker info differently since we're using raw components
            from src.reranker import Reranker
            use_rr = mode

            results = run_test(doc_name, doc_text, target_qs, config, embedder, llm, use_rr)

            avg_sem = sum(r["semantic_sim"] for r in results) / len(results)
            avg_llm = sum(r["avg"] for r in results) / len(results)

            print(f"  avg sem_sim: {avg_sem:.3f}  avg LLM: {avg_llm:.1f}")
            for r in results:
                print(f"  Q{r['q_idx']+1}: sem={r['semantic_sim']:.3f} "
                      f"acc={r['scores'].get('accuracy',0)} comp={r['scores'].get('completeness',0)} "
                      f"conc={r['scores'].get('conciseness',0)} avg={r['avg']}")
            print(f"  time: {time.time()-t0:.0f}s")

            all_results.append({
                "doc": doc_name, "config": f"{cs}/{ov}/{tk}",
                "reranker": label, "avg_sem_sim": round(avg_sem, 3),
                "avg_llm": round(avg_llm, 1),
                "details": results,
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Reranker OFF vs ON")
    print("=" * 60)

    for i in range(0, len(all_results), 2):
        off = all_results[i]
        on = all_results[i + 1]
        sem_gain = on["avg_sem_sim"] - off["avg_sem_sim"]
        llm_gain = on["avg_llm"] - off["avg_llm"]
        print(f"\n{off['doc']} ({off['config']}):")
        print(f"  sem_sim: {off['avg_sem_sim']:.3f} -> {on['avg_sem_sim']:.3f} ({sem_gain:+.3f})")
        print(f"  LLM:     {off['avg_llm']:.1f} -> {on['avg_llm']:.1f} ({llm_gain:+.1f})")

    # Save
    with open(f"{EVAL_DIR}/reranker_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: reranker_results.json")


if __name__ == "__main__":
    main()
