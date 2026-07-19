import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from threading import Lock

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "documents"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY, stream_usage=True
)

os.makedirs("logs", exist_ok=True)
chat_logger = logging.getLogger("rag.chat")
chat_logger.setLevel(logging.INFO)
if not chat_logger.handlers:
    _handler = logging.FileHandler(os.path.join("logs", "app.log"), encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(message)s"))
    chat_logger.addHandler(_handler)
chat_logger.propagate = False

_metrics_lock = Lock()
_metrics = {
    "chat_requests": 0,
    "errors": 0,
    "total_retrieval_ms": 0.0,
    "total_llm_ms": 0.0,
    "total_latency_ms": 0.0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
}


def _record_chat(record: dict):
    chat_logger.info(json.dumps(record))
    with _metrics_lock:
        _metrics["chat_requests"] += 1
        if record.get("status") == "error":
            _metrics["errors"] += 1
        _metrics["total_retrieval_ms"] += record.get("retrieval_ms", 0.0)
        _metrics["total_llm_ms"] += record.get("llm_ms", 0.0)
        _metrics["total_latency_ms"] += record.get("total_ms", 0.0)
        _metrics["total_input_tokens"] += record.get("input_tokens", 0)
        _metrics["total_output_tokens"] += record.get("output_tokens", 0)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)

# Chunks scoring below this cosine relevance are dropped from the LLM prompt
# and the sources list; if nothing passes, the app refuses without calling the LLM.
MIN_RELEVANCE = 0.25

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "context extracted from uploaded documents. If the context does not contain "
    "the information needed to answer the question, respond exactly with: "
    "\"I don't have that information in the uploaded documents\". "
    "Do not use any outside knowledge. Be concise and accurate. "
    "Earlier conversation turns may be provided; use them only to understand "
    "what the user is referring to, never as a source of facts."
)

app = FastAPI(title="RAG Chatbot")


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryMessage] = []


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int


def _known_filenames() -> list[str]:
    data = vectorstore.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    seen = set()
    for meta in metadatas:
        if meta and meta.get("filename"):
            seen.add(meta["filename"])
    return sorted(seen)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/documents")
async def list_documents():
    return {"filenames": _known_filenames()}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.pdf")
    try:
        with open(tmp_path, "wb") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)

        reader = PdfReader(tmp_path)
        if len(reader.pages) == 0:
            raise HTTPException(status_code=400, detail="The PDF has no pages.")

        raw_docs = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                raw_docs.append(
                    Document(
                        page_content=page_text,
                        metadata={"filename": file.filename, "page": page_number},
                    )
                )

        if not raw_docs:
            raise HTTPException(
                status_code=400, detail="No extractable text found in the PDF."
            )

        chunks = text_splitter.split_documents(raw_docs)
        for chunk in chunks:
            chunk.metadata["filename"] = chunk.metadata.get("filename", file.filename)
            chunk.metadata["page"] = chunk.metadata.get("page", 1)

        vectorstore.add_documents(chunks)

        return UploadResponse(filename=file.filename, chunks_added=len(chunks))
    finally:
        file.file.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/metrics")
async def metrics():
    with _metrics_lock:
        snapshot = dict(_metrics)
    n = snapshot["chat_requests"] or 1
    return {
        "chat_requests": snapshot["chat_requests"],
        "errors": snapshot["errors"],
        "total_input_tokens": snapshot["total_input_tokens"],
        "total_output_tokens": snapshot["total_output_tokens"],
        "total_tokens": snapshot["total_input_tokens"]
        + snapshot["total_output_tokens"],
        "avg_retrieval_ms": round(snapshot["total_retrieval_ms"] / n, 1),
        "avg_llm_ms": round(snapshot["total_llm_ms"] / n, 1),
        "avg_latency_ms": round(snapshot["total_latency_ms"] / n, 1),
    }


@app.delete("/documents")
async def clear_documents():
    data = vectorstore.get()
    ids = data.get("ids") or []
    if ids:
        vectorstore.delete(ids=ids)
    return {"cleared": len(ids)}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _chunk_text(chunk) -> str:
    text = chunk.text
    if callable(text):
        text = text()
    return text or ""


NO_INFO_ANSWER = "I don't have that information in the uploaded documents"


@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    history = [
        {"role": m.role, "content": m.content}
        for m in request.history[-12:]
        if m.role in ("user", "assistant") and m.content.strip()
    ]

    def event_stream():
        t_start = time.perf_counter()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": "/chat",
            "question_chars": len(question),
            "history_messages": len(history),
            "retrieval_ms": 0.0,
            "llm_ms": 0.0,
            "total_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "status": "ok",
        }

        def finish(status: str):
            record["status"] = status
            record["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            _record_chat(record)

        try:
            t0 = time.perf_counter()
            results = vectorstore.similarity_search_with_relevance_scores(
                question, k=4
            )
            record["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        except Exception:
            finish("error")
            yield _sse(
                {
                    "type": "error",
                    "message": "Searching the document store failed. Please try again.",
                }
            )
            return

        results = [(doc, score) for doc, score in results if score >= MIN_RELEVANCE]

        if not results:
            finish("no_results")
            yield _sse({"type": "sources", "sources": []})
            yield _sse({"type": "token", "token": NO_INFO_ANSWER})
            yield _sse({"type": "done"})
            return

        context_blocks = []
        sources = []
        for doc, score in results:
            filename = doc.metadata.get("filename", "unknown")
            page = doc.metadata.get("page", 0)
            context_blocks.append(
                f"[Source: {filename}, page {page}]\n{doc.page_content}"
            )
            sources.append(
                {"filename": filename, "page": page, "score": round(score, 4)}
            )

        yield _sse({"type": "sources", "sources": sources})

        context_text = "\n\n---\n\n".join(context_blocks)
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_text}\n\n"
                        f"Question: {question}\n\n"
                        "Answer using only the context above."
                    ),
                }
            ]
        )

        try:
            t1 = time.perf_counter()
            for chunk in llm.stream(messages):
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    record["input_tokens"] = usage.get("input_tokens", 0)
                    record["output_tokens"] = usage.get("output_tokens", 0)
                    record["total_tokens"] = usage.get("total_tokens", 0)
                text = _chunk_text(chunk)
                if text:
                    yield _sse({"type": "token", "token": text})
            record["llm_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        except Exception:
            finish("error")
            yield _sse(
                {
                    "type": "error",
                    "message": "The AI service did not respond. Please try again.",
                }
            )
            return

        finish("ok")
        yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/static", StaticFiles(directory="static"), name="static")
