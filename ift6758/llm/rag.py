from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import requests

DOCS_URL = "https://raw.githubusercontent.com/Zmalski/NHL-API-Reference/main/README.md"

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
LLM_DIR= Path(__file__).resolve().parent / "llm_docs"



def load_docs(llm_dir=LLM_DIR):
    doc_path = Path(llm_dir)/ "nhl-api-reference.md"
    if not doc_path.exists():
        response = requests.get(DOCS_URL, timeout=30)
        response.raise_for_status()
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(response.text, encoding="utf-8")

    text = doc_path.read_text(encoding="utf-8")
    web_docs = text.split("# NHL Web API Documentation", 1)[1]
    return web_docs.split("# NHL Stats API Documentation", 1)[0]


def chunker(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks


def build_index(text, chunk_size=800, chunk_overlap=120):
    embedder = SentenceTransformer(EMBEDDER, device="cpu")
    chunks=chunker(text, chunk_size, chunk_overlap)
    vectors = embedder.encode(chunks, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # dot product of unit vectors = cosine similarity
    index.add(vectors)
    return chunks, embedder, index


def retrieve(query,chunks,embedder,index,k=3):
    query_vector = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, positions = index.search(query_vector, k)
    context = "\n\n".join(f"[{i}] {chunks[position]}" for i, position in enumerate(positions[0], 1))
    return context

def save_retrieval(path, query, context):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Question\n\n{query}\n\n# Retrieved exerpts\n\n{context}\n",encoding="utf-8")
   