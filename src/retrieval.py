"""
Hybrid search + reranking.

"Hybrid" means every query is answered by two independent retrievers, fused
together:
  - a DENSE retriever (Chroma + local embeddings) that matches on semantic
    similarity, good at paraphrases and conceptual matches;
  - a SPARSE retriever (BM25) that matches on exact keyword/token overlap,
    good at IDs, error codes, and exact terminology dense embeddings blur.

`EnsembleRetriever` fuses their two ranked lists (Reciprocal Rank Fusion)
into one candidate set, weighted by DENSE_WEIGHT / SPARSE_WEIGHT.

That fused list is then passed through a cross-encoder RERANKER
(`ContextualCompressionRetriever` + `FlashrankRerank`), which scores each
(query, document) pair jointly instead of comparing independent embeddings.
Cross-encoders are much more accurate at judging relevance than the dense/
sparse retrievers, but too slow to run over a whole corpus - so they only
re-score the small candidate set the hybrid step already narrowed down, and
we keep just the top RERANK_TOP_N.

Two extras for document Q&A:
  - Each file's metadata (title, author, ...) is indexed as its own small
    chunk, so questions like "who is the author?" have something to match.
  - A short description of what the indexed documents cover is written at
    index time (kb_description.txt) and read by the router in nodes.py, so
    it can tell whether a question is about the user's documents.

Embeddings run locally via HuggingFace/sentence-transformers (HF_TOKEN is
wired in for gated models / higher Hub rate limits, see config.py).
Reranking runs locally via FlashRank (ONNX, no token needed). The KB
description uses the grading LLM (needs its API key) but falls back to raw
excerpts if that call fails.
"""
import functools
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

# extension -> function that builds a loader for a file path
SUPPORTED_EXTENSIONS = {
    ".pdf": lambda p: PyMuPDFLoader(p),
    ".docx": lambda p: Docx2txtLoader(p),
    ".txt": lambda p: TextLoader(p, encoding="utf-8", autodetect_encoding=True),
    ".md": lambda p: TextLoader(p, encoding="utf-8", autodetect_encoding=True),
    ".csv": lambda p: CSVLoader(p, encoding="utf-8"),
}

# Metadata fields worth indexing as a searchable chunk
_METADATA_FIELDS = ("title", "author", "subject", "keywords", "creator", "producer")


def _metadata_chunk(path: Path, metadata: dict, page_count: int):
    """Build a small Document describing the file itself, or None if empty."""
    info = ", ".join(
        f"{k}: {metadata[k]}" for k in _METADATA_FIELDS if str(metadata.get(k, "")).strip()
    )
    if not info:
        return None
    return Document(
        page_content=(
            f"Document '{path.name}' metadata. {info}. "
            f"Number of pages/sections: {page_count}."
        ),
        metadata={"source": path.name, "is_metadata": True},
    )


def load_and_split_documents(data_dir: str = None):
    """Load every supported file in data_dir and split it into overlapping chunks."""
    data_dir = data_dir or config.DATA_DIR
    print(f"[load] scanning {Path(data_dir).resolve()}")
    docs = []
    metadata_docs = []
    for path in sorted(Path(data_dir).rglob("*")):
        make_loader = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
        if not path.is_file() or make_loader is None:
            continue
        try:
            loaded = make_loader(str(path)).load()
        except Exception as e:
            print(f"[load] Skipping {path.name}: {type(e).__name__}: {e}")
            continue

        # Capture file-level metadata before dropping any pages
        meta_doc = _metadata_chunk(path, loaded[0].metadata, len(loaded)) if loaded else None

        # Drop pages with no extractable text (blank or scanned pages)
        loaded = [d for d in loaded if d.page_content.strip()]
        if not loaded:
            print(f"[load] {path.name}: no extractable text (scanned PDF? needs OCR)")
            continue

        for d in loaded:
            d.metadata["source"] = path.name
        print(f"[load] {path.name}: {len(loaded)} page(s)/section(s)")
        docs.extend(loaded)
        if meta_doc:
            metadata_docs.append(meta_doc)

    if not docs:
        raise FileNotFoundError(
            f"No readable documents ({', '.join(SUPPORTED_EXTENSIONS)}) found in "
            f"{data_dir!r}. Add some source docs, or point DATA_DIR at a "
            "directory that has them."
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    # Metadata chunks are already small, so they are added after splitting
    return splitter.split_documents(docs) + metadata_docs


# ----------------------------------------------------- knowledge-base description

KB_DESCRIPTION_FILE = "kb_description.txt"


def _kb_description_path(persist_directory: str = None) -> Path:
    return Path(persist_directory or config.CHROMA_DIR) / KB_DESCRIPTION_FILE


def build_kb_description(chunks, chunks_per_file: int = 5, snippet_chars: int = 300) -> str:
    """
    Describe what the indexed documents cover, so the router can judge whether
    a question is about them. Samples a few chunks per file and asks the
    grading LLM for a short topic summary. If that call fails, the raw
    excerpts are used instead.
    """
    by_source = {}
    for c in chunks:
        if c.metadata.get("is_metadata"):
            continue  # sample real content, not the metadata chunk
        by_source.setdefault(c.metadata.get("source", "unknown"), []).append(c)

    samples = []
    for source, file_chunks in by_source.items():
        step = max(1, len(file_chunks) // chunks_per_file)
        picked = file_chunks[::step][:chunks_per_file]
        excerpts = " ... ".join(
            " ".join(c.page_content.split())[:snippet_chars] for c in picked
        )
        samples.append(f"File: {source}\nExcerpts: {excerpts}")
    sample_text = "\n\n".join(samples)

    try:
        # Local imports: only needed at index time
        from langchain_core.output_parsers import StrOutputParser

        from .llm import get_grading_llm

        prompt = (
            "Below are excerpts from each file in a user's document collection. "
            "Write one line per file: the file name, then the main topics and key "
            "terms it covers (at most 25 words per file). Return only those lines.\n\n"
            + sample_text
        )
        description = (get_grading_llm() | StrOutputParser()).invoke(prompt).strip()
        if not description:
            raise ValueError("empty description")
    except Exception as e:
        print(f"[kb] LLM summary failed ({type(e).__name__}: {e}); using raw excerpts")
        description = sample_text
    return description[:2000]


def save_kb_description(chunks, persist_directory: str = None) -> str:
    description = build_kb_description(chunks)
    path = _kb_description_path(persist_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(description, encoding="utf-8")
    return description


def load_kb_description(persist_directory: str = None) -> str:
    """Read the saved description; the router calls this on every question."""
    try:
        text = _kb_description_path(persist_directory).read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    return text or "(no description available)"


# ------------------------------------------------------------- vector store


def get_embeddings():
    # Local import: only pulled in when embeddings are actually needed
    # (needs `sentence-transformers`, see requirements.txt).
    from langchain_huggingface import HuggingFaceEmbeddings

    model_kwargs = {"token": config.HF_TOKEN} if config.HF_TOKEN else {}
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, model_kwargs=model_kwargs)


def build_vectorstore(chunks, persist_directory: str = None):
    """
    Embed `chunks` and persist them to a local Chroma collection. Run via
    ingest.py or the Streamlit upload button. Full rebuild: any existing
    collection is dropped first, so re-indexing never leaves duplicate or
    stale chunks behind. Also writes the KB description the router uses.
    """
    persist_directory = persist_directory or config.CHROMA_DIR
    embeddings = get_embeddings()
    Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    ).delete_collection()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=config.COLLECTION_NAME,
    )
    save_kb_description(chunks, persist_directory)
    return store


def load_vectorstore(persist_directory: str = None):
    persist_directory = persist_directory or config.CHROMA_DIR
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
    )


# ---------------------------------------------------------- hybrid retrieval


def get_dense_retriever():
    """The semantic half of hybrid search."""
    return load_vectorstore().as_retriever(search_kwargs={"k": config.DENSE_K})


def get_sparse_retriever(chunks=None):
    """The keyword half of hybrid search."""
    chunks = chunks if chunks is not None else load_and_split_documents()
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = config.SPARSE_K
    return retriever


def get_hybrid_retriever():
    """Fuse dense + sparse retrievers via weighted Reciprocal Rank Fusion."""
    chunks = load_and_split_documents()
    return EnsembleRetriever(
        retrievers=[get_dense_retriever(), get_sparse_retriever(chunks)],
        weights=[config.DENSE_WEIGHT, config.SPARSE_WEIGHT],
    )


def get_reranker():
    """Cross-encoder reranker applied on top of the hybrid candidate set."""
    # Local imports: only pulled in when reranking is actually needed.
    from flashrank import Ranker
    from langchain_community.document_compressors import FlashrankRerank

    # FlashRank defaults its model cache to /tmp, which is cleared on reboot
    # (and on every fresh container), forcing a re-download - so pass an
    # explicit, persistent cache_dir instead.
    ranker = Ranker(model_name=config.RERANKER_MODEL, cache_dir=config.RERANKER_CACHE_DIR)
    return FlashrankRerank(client=ranker, top_n=config.RERANK_TOP_N)


@functools.lru_cache(maxsize=1)
def get_retriever():
    """
    The full pipeline used by the graph: hybrid (dense + sparse) search,
    then cross-encoder reranking. Cached per-process so the BM25 index and
    ONNX models are only built once.
    """
    return ContextualCompressionRetriever(
        base_compressor=get_reranker(),
        base_retriever=get_hybrid_retriever(),
    )