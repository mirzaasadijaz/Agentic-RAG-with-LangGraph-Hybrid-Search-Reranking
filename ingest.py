"""
Build the local hybrid-search index from DATA_DIR. Run this once (and again
any time the source documents change) before using the app. Safe to re-run:
each run fully rebuilds the index rather than adding to it.

Usage:
    python ingest.py
"""
from src import config
from src.retrieval import build_vectorstore, load_and_split_documents


def main():
    print(f"Loading documents from {config.DATA_DIR} ...")
    chunks = load_and_split_documents(config.DATA_DIR)
    print(f"Split into {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    print(f"Rebuilding Chroma index at {config.CHROMA_DIR} "
          f"(model={config.EMBEDDING_MODEL}, first run downloads the model) ...")
    build_vectorstore(chunks, persist_directory=config.CHROMA_DIR)

    print("Done. The BM25 (sparse) index needs no separate build step - it's "
          "rebuilt in memory from DATA_DIR each time the app starts.")
    print("\nYou can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()