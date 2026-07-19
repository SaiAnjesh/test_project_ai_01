"""Evaluation suite for the RAG chatbot pipeline.

Generates 10 Q/A pairs from test_document.pdf (cached in eval_questions.json),
runs each through retrieval + generation, and reports:
  - retrieval hit rate in top-4 (verbatim evidence found in a retrieved chunk)
  - faithfulness 1-5 scored by gpt-4o-mini with a fixed rubric at temperature 0
  - average latency and token usage

Usage:
  .venv/Scripts/python.exe eval.py [--reingest] [--regenerate]
"""

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from main import (
    FALLBACK_CHUNKS,
    MIN_RELEVANCE,
    NO_INFO_ANSWER,
    OPENAI_API_KEY,
    SYSTEM_PROMPT,
    llm,
    text_splitter,
    vectorstore,
)

TEST_PDF = "test_document.pdf"
QUESTIONS_FILE = "eval_questions.json"
RESULTS_FILE = "eval_results.json"
NUM_QUESTIONS = 10
TOP_K = 4

judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

JUDGE_RUBRIC = """You are grading the faithfulness of an answer against retrieved context.
Use this fixed rubric and reply with a single integer from 1 to 5, nothing else:
5 - Every factual claim in the answer is directly supported by the context.
4 - The answer is supported by the context with only trivial unsupported phrasing.
3 - The answer is partially supported; some claims are not in the context.
2 - The answer is mostly unsupported by the context.
1 - The answer contradicts the context or is fabricated.
Special case: if the answer states that the information is not available, score 5
if the context truly lacks the answer, and 1 if the context clearly contains it."""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_pdf_pages() -> list[Document]:
    reader = PdfReader(TEST_PDF)
    docs = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            docs.append(
                Document(
                    page_content=page_text,
                    metadata={"filename": TEST_PDF, "page": page_number},
                )
            )
    return docs


def ensure_ingested(force: bool = False) -> int:
    data = vectorstore.get(include=["metadatas"])
    ids = data.get("ids") or []
    metadatas = data.get("metadatas") or []
    test_ids = [
        id_
        for id_, meta in zip(ids, metadatas)
        if meta and meta.get("filename") == TEST_PDF
    ]
    if test_ids and not force:
        return len(test_ids)
    if test_ids:
        vectorstore.delete(ids=test_ids)
    chunks = text_splitter.split_documents(load_pdf_pages())
    vectorstore.add_documents(chunks)
    return len(chunks)


def generate_questions() -> list[dict]:
    pages = load_pdf_pages()
    doc_text = "\n\n".join(
        f"[Page {d.metadata['page']}]\n{d.page_content}" for d in pages
    )
    full_norm = normalize(doc_text)

    prompt = f"""Below is the full text of a document, with page markers.

{doc_text}

Create {NUM_QUESTIONS + 4} factual question/answer pairs that can each be answered
from a single specific place in the document. Spread them across all pages.
Return ONLY a JSON array. Each element must have exactly these keys:
- "question": a natural question a user might ask
- "expected_answer": the correct short answer
- "evidence": a VERBATIM quote of 5-15 consecutive words copied exactly from the
  document text above that contains the answer
- "page": the page number the evidence appears on (integer)
Do not paraphrase the evidence; copy it character for character."""

    response = judge_llm.invoke(
        [{"role": "user", "content": prompt}]
    )
    text = response.content
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise RuntimeError("Question generator did not return a JSON array.")
    candidates = json.loads(match.group(0))

    valid = []
    for c in candidates:
        if not all(k in c for k in ("question", "expected_answer", "evidence", "page")):
            continue
        if normalize(c["evidence"]) in full_norm:
            valid.append(c)
        if len(valid) == NUM_QUESTIONS:
            break
    if len(valid) < NUM_QUESTIONS:
        raise RuntimeError(
            f"Only {len(valid)} of {len(candidates)} generated questions had "
            "verbatim evidence; re-run to try again."
        )
    return valid


def run_pipeline(question: str) -> dict:
    t0 = time.perf_counter()
    results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Same prompt filter as the app: hit rate is judged on the raw top-4,
    # but only chunks above MIN_RELEVANCE are sent to the LLM (with the same
    # top-2 fallback for questions that score low against every chunk).
    prompt_results = [(doc, score) for doc, score in results if score >= MIN_RELEVANCE]
    if not prompt_results and results:
        prompt_results = results[:FALLBACK_CHUNKS]
    context_blocks = [
        f"[Source: {doc.metadata.get('filename', '?')}, page {doc.metadata.get('page', '?')}]\n"
        f"{doc.page_content}"
        for doc, _ in prompt_results
    ]
    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "(empty)"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n\n"
                "Answer using only the context above."
            ),
        },
    ]
    t1 = time.perf_counter()
    response = llm.invoke(messages)
    llm_ms = (time.perf_counter() - t1) * 1000

    usage = response.usage_metadata or {}
    return {
        "answer": response.content,
        "context_text": context_text,
        "retrieved": [(doc.page_content, doc.metadata, score) for doc, score in results],
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
        "total_ms": retrieval_ms + llm_ms,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


def judge_faithfulness(question: str, context_text: str, answer: str) -> int:
    response = judge_llm.invoke(
        [
            {"role": "system", "content": JUDGE_RUBRIC},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer: {answer}\n\n"
                    "Score (1-5):"
                ),
            },
        ]
    )
    match = re.search(r"[1-5]", response.content)
    return int(match.group(0)) if match else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reingest", action="store_true", help="re-chunk and re-embed the test PDF")
    parser.add_argument("--regenerate", action="store_true", help="regenerate eval questions")
    args = parser.parse_args()

    chunk_count = ensure_ingested(force=args.reingest)
    print(f"Test PDF ingested: {chunk_count} chunks "
          f"(chunk_size={text_splitter._chunk_size}, overlap={text_splitter._chunk_overlap})")

    if args.regenerate or not _questions_exist():
        print("Generating eval questions with gpt-4o-mini...")
        questions = generate_questions()
        with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2)
        print(f"Saved {len(questions)} questions to {QUESTIONS_FILE}")
    else:
        with open(QUESTIONS_FILE, encoding="utf-8") as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions from {QUESTIONS_FILE}")

    rows = []
    for i, q in enumerate(questions, 1):
        result = run_pipeline(q["question"])
        hit = any(
            normalize(q["evidence"]) in normalize(chunk_text)
            for chunk_text, _, _ in result["retrieved"]
        )
        faith = judge_faithfulness(q["question"], result["context_text"], result["answer"])
        rows.append(
            {
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "evidence_page": q["page"],
                "answer": result["answer"],
                "hit": hit,
                "faithfulness": faith,
                "retrieval_ms": round(result["retrieval_ms"], 1),
                "llm_ms": round(result["llm_ms"], 1),
                "total_ms": round(result["total_ms"], 1),
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
            }
        )
        print(f"  [{i}/{len(questions)}] hit={'Y' if hit else 'N'} faith={faith} "
              f"{q['question'][:60]}")

    hit_rate = sum(r["hit"] for r in rows) / len(rows)
    summary = {
        "hit_rate_top4": round(hit_rate, 3),
        "avg_faithfulness": round(statistics.mean(r["faithfulness"] for r in rows), 2),
        "avg_latency_ms": round(statistics.mean(r["total_ms"] for r in rows), 1),
        "avg_retrieval_ms": round(statistics.mean(r["retrieval_ms"] for r in rows), 1),
        "avg_llm_ms": round(statistics.mean(r["llm_ms"] for r in rows), 1),
        "avg_input_tokens": round(statistics.mean(r["input_tokens"] for r in rows), 1),
        "avg_output_tokens": round(statistics.mean(r["output_tokens"] for r in rows), 1),
    }

    print()
    print("=" * 96)
    print(f"{'#':<3}{'Question':<48}{'Hit':<5}{'Faith':<7}{'ms':<8}{'in tok':<8}{'out tok':<8}")
    print("-" * 96)
    for i, r in enumerate(rows, 1):
        print(f"{i:<3}{r['question'][:46]:<48}{'Y' if r['hit'] else 'N':<5}"
              f"{r['faithfulness']:<7}{r['total_ms']:<8.0f}{r['input_tokens']:<8}{r['output_tokens']:<8}")
    print("-" * 96)
    print(f"Hit rate (top-{TOP_K}): {hit_rate:.0%}   "
          f"Avg faithfulness: {summary['avg_faithfulness']}/5   "
          f"Avg latency: {summary['avg_latency_ms']:.0f} ms   "
          f"Avg tokens: {summary['avg_input_tokens']:.0f} in / {summary['avg_output_tokens']:.0f} out")
    print("=" * 96)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "chunk_size": text_splitter._chunk_size,
                    "chunk_overlap": text_splitter._chunk_overlap,
                    "top_k": TOP_K,
                    "embedding_model": "text-embedding-3-small",
                    "llm": "gpt-4o-mini",
                    "chunks_in_store": chunk_count,
                },
                "summary": summary,
                "per_question": rows,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {RESULTS_FILE}")


def _questions_exist() -> bool:
    try:
        with open(QUESTIONS_FILE, encoding="utf-8") as f:
            return len(json.load(f)) >= NUM_QUESTIONS
    except (OSError, ValueError):
        return False


if __name__ == "__main__":
    sys.exit(main())
