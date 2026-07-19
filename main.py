import os
import shutil
import tempfile
import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
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
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "context extracted from uploaded documents. If the context does not contain "
    "the information needed to answer the question, respond exactly with: "
    "\"I don't have that information in the uploaded documents\". "
    "Do not use any outside knowledge. Be concise and accurate."
)

app = FastAPI(title="RAG Chatbot")


class ChatRequest(BaseModel):
    message: str


class SourceInfo(BaseModel):
    filename: str
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    results = vectorstore.similarity_search_with_relevance_scores(question, k=4)

    if not results:
        return ChatResponse(
            answer="I don't have that information in the uploaded documents",
            sources=[],
        )

    context_blocks = []
    sources = []
    for doc, score in results:
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", 0)
        context_blocks.append(
            f"[Source: {filename}, page {page}]\n{doc.page_content}"
        )
        sources.append(SourceInfo(filename=filename, page=page, score=round(score, 4)))

    context_text = "\n\n---\n\n".join(context_blocks)

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

    response = llm.invoke(messages)
    answer = response.content

    return ChatResponse(answer=answer, sources=sources)


app.mount("/static", StaticFiles(directory="static"), name="static")
