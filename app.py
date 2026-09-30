"""
Streamlit UI for the agentic RAG system.

Answers render like a paper: LaTeX equations (every common delimiter style),
syntax-highlighted code blocks with copy buttons and line numbers, tables and
lists. Each agent node appears live in a timeline while it runs; afterwards the
answer keeps its sources, steps and raw markdown. Upload your own documents from
the sidebar to add them to the knowledge base.

Usage:
    streamlit run app.py
"""
import html
import re
import time
import traceback
from pathlib import Path

import streamlit as st

from src import config
from src.graph import build_graph
from src.retrieval import (
    SUPPORTED_EXTENSIONS,
    build_vectorstore,
    get_retriever,
    load_and_split_documents,
)

INITIAL_STATE = {
    "documents": [],
    "generation": "",
    "web_search": "No",
    "retry_count": 0,
}

# Shown on the empty screen. Edit these to match your documents.
EXAMPLES = [
    "Summarize the key ideas in my documents",
    "What are the main topics covered?",
    "Explain the most important concept in simple terms",
    "What are the key takeaways I should remember?",
]

LANG_ALIASES = {
    "py": "python", "js": "javascript", "ts": "typescript", "sh": "bash",
    "shell": "bash", "zsh": "bash", "yml": "yaml", "c++": "cpp",
}

# Serif answers match KaTeX's serif math. Sans and mono come from Streamlit's own fonts.
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');
:root {
  --serif: 'Source Serif 4', Georgia, 'Times New Roman', serif;
  --sans: 'Source Sans', 'Source Sans Pro', 'Source Sans 3', system-ui, sans-serif;
  --line: rgba(128, 128, 128, .30);
  --soft: rgba(128, 128, 128, .10);
  --ink: #4f7cac;
}
.block-container, [data-testid="stMainBlockContainer"] { max-width: 820px !important; padding-top: 4rem; }
footer { visibility: hidden; }
.stButton > button, .stDownloadButton > button { width: 100%; }

.x-title { font: 600 2rem/1.2 var(--serif); letter-spacing: -.01em; }
.x-lede { margin: .3rem 0 1.5rem; font: 1rem/1.5 var(--sans); opacity: .65; }

/* Conversation: questions sit in a soft block, answers read as open prose */
[data-testid="stChatMessage"] { padding: .6rem 0 !important; background: transparent !important; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { font: 1.06rem/1.68 var(--serif); }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] :is(p, li) { max-width: 72ch; }
[data-testid="stChatMessage"] :is(summary, [data-baseweb="tab"], button) [data-testid="stMarkdownContainer"] { font: .92rem/1.4 var(--sans); }
[data-testid="stChatMessage"] :is(h1, h2, h3, h4) { font-family: var(--serif); font-weight: 600; letter-spacing: 0; padding: .6rem 0 .1rem; }
[data-testid="stChatMessage"] h1 { font-size: 1.5rem; }
[data-testid="stChatMessage"] h2 { font-size: 1.28rem; }
[data-testid="stChatMessage"] h3 { font-size: 1.12rem; }
[data-testid="stChatMessage"] h4 { font-size: 1rem; }
@supports selector(:has(*)) {
  [data-testid="stChatMessage"] :is([data-testid^="stChatMessageAvatar"], [data-testid^="chatAvatarIcon"]) { display: none; }
  [data-testid="stChatMessage"]:has(:is([data-testid="stChatMessageAvatarUser"], [data-testid="chatAvatarIcon-user"])) { padding: .7rem 1rem !important; background: var(--soft) !important; border-radius: 12px; }
  [data-testid="stChatMessage"]:has(:is([data-testid="stChatMessageAvatarUser"], [data-testid="chatAvatarIcon-user"])) [data-testid="stMarkdownContainer"] { font: 1rem/1.5 var(--sans); }
}

/* Tables, quotes, inline code, code boxes, equations */
[data-testid="stMarkdownContainer"] table { display: block; max-width: 100%; overflow-x: auto; border-collapse: collapse; font-size: .95em; font-variant-numeric: tabular-nums; }
[data-testid="stMarkdownContainer"] :is(th, td) { padding: .4rem .8rem; border: 1px solid var(--line); }
[data-testid="stMarkdownContainer"] th { background: var(--soft); text-align: left; font-weight: 600; }
[data-testid="stMarkdownContainer"] blockquote { margin-left: 0; padding: .05rem 1rem; border-left: 3px solid var(--ink); }
[data-testid="stMarkdownContainer"] :not(pre) > code { padding: .1em .35em; border-radius: 4px; background: var(--soft); color: inherit; }
[data-testid="stCode"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.katex-display { overflow-x: auto; overflow-y: hidden; padding: .25rem 0; }

/* Answer footer: timing line, agent timeline, numbered sources */
.x-meta { margin-top: .4rem; font: .84rem/1.4 var(--sans); opacity: .6; }
.x-tl { margin: .2rem 0 .2rem .35rem; padding-left: 1.1rem; border-left: 1px solid var(--line); font: .9rem/1.4 var(--sans); }
.x-tl .s { position: relative; display: flex; flex-wrap: wrap; gap: .2rem .7rem; align-items: baseline; padding: .3rem 0; }
.x-tl .s::before { content: ""; position: absolute; left: -1.35rem; top: .75rem; width: .5rem; height: .5rem; border-radius: 50%; background: var(--ink); }
.x-tl .s span { font-size: .84rem; opacity: .62; }
.x-src { margin: .6rem 0 1rem; padding-left: .85rem; border-left: 2px solid var(--line); }
.x-src .h { font: 600 .88rem/1.4 var(--sans); }
.x-src .h i { font-style: normal; font-weight: 400; opacity: .55; }
.x-src .t { margin-top: .15rem; font: .92rem/1.55 var(--serif); opacity: .78; }
.x-files div { padding: .3rem 0; border-bottom: 1px solid var(--line); font-size: .86rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>"""

_CODE_OR_MATH = re.compile(r"(```[^\n]*\n.*?```|\$\$.+?\$\$)", re.S)
_CODE_SPANS = re.compile(r"(```.*?```|`[^`\n]+`)", re.S)


# ---------------------------------------------------------------- backend glue
@st.cache_resource(show_spinner="Building agent graph...")
def get_graph():
    config.check_required_api_keys()
    return build_graph()


@st.cache_data(show_spinner=False)
def get_graph_png():
    """Render the graph diagram (needs internet for mermaid.ink). None on failure."""
    try:
        return get_graph().get_graph().draw_mermaid_png()
    except Exception:
        return None


def run_agent(graph, question: str):
    """Yield (node_name, update_dict) for every node that runs."""
    state = {**INITIAL_STATE, "question": question, "original_question": question}
    for step in graph.stream(state, stream_mode="updates"):
        for node_name, update in step.items():
            yield node_name, (update or {})


def list_indexed_files():
    data_dir = Path(config.DATA_DIR)
    if not data_dir.exists():
        return []
    return sorted(
        p.name for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# ------------------------------------------------------------ answer rendering
def normalize_math(text):
    """Rewrite \\( \\) and \\[ \\] math (what most LLMs emit) as $ and $$, skipping code."""
    def fix(chunk):
        chunk = re.sub(r"(?<!\\)\\\[(.+?)\\\]", lambda m: f"\n\n$${m.group(1).strip()}$$\n\n", chunk, flags=re.S)
        return re.sub(r"(?<!\\)\\\((.+?)\\\)", lambda m: f"${m.group(1).strip()}$", chunk, flags=re.S)

    return "".join(p if i % 2 else fix(p) for i, p in enumerate(_CODE_SPANS.split(text)))


def show_code(code, lang=None):
    try:
        st.code(code, language=lang, line_numbers=code.count("\n") >= 8)
    except TypeError:  # older Streamlit without line_numbers
        st.code(code, language=lang)


def render_answer(text):
    """Markdown, with display equations as st.latex and code fences as st.code boxes."""
    for i, part in enumerate(_CODE_OR_MATH.split(normalize_math(text))):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        elif part.startswith("```"):
            head, _, body = part[3:].partition("\n")
            lang = (head.split() or [""])[0].lower()
            show_code(body.rsplit("```", 1)[0].rstrip("\n"), LANG_ALIASES.get(lang, lang) or None)
        else:
            st.latex(part.strip("$").strip())


# ----------------------------------------------------- steps, sources, details
def plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def describe_step(node, update, question):
    """Turn one node update into a label plus a short plain-language detail."""
    bits = []
    if "documents" in update:
        bits.append(plural(len(update["documents"] or []), "document"))
    if "web_search" in update:
        needed = str(update["web_search"]).lower() == "yes"
        bits.append("web search needed" if needed else "web search not needed")
    if update.get("retry_count"):
        bits.append(f"retry {update['retry_count']}")
    if update.get("question") and update["question"] != question:
        bits.append(f"rewrote question as “{update['question']}”")
    if update.get("generation"):
        bits.append(plural(len(str(update["generation"]).split()), "word"))
    return {"label": node.replace("_", " ").capitalize(), "detail": ", ".join(bits)}


def timeline_html(trace):
    rows = "".join(
        f'<div class="s"><b>{html.escape(s["label"])}</b>'
        + (f'<span>{html.escape(s["detail"])}</span>' if s["detail"] else "")
        + "</div>"
        for s in trace
    )
    return f'<div class="x-tl">{rows}</div>'


def to_source(doc):
    """Reduce a retrieved document (LangChain Document, dict or str) to what the UI shows."""
    if isinstance(doc, dict):
        text, meta = doc.get("page_content") or doc.get("content") or "", doc.get("metadata")
    else:
        text, meta = getattr(doc, "page_content", str(doc)), getattr(doc, "metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    name = str(meta.get("source") or meta.get("url") or meta.get("title") or "Untitled source")
    page = meta.get("page")
    text = " ".join(str(text).split())
    return {
        "name": name if name.startswith(("http://", "https://")) else Path(name).name,
        "page": page + 1 if isinstance(page, int) else None,
        "text": text[:500] + ("…" if len(text) > 500 else ""),
    }


def source_html(i, s):
    name = html.escape(s["name"])
    if s["name"].startswith(("http://", "https://")):
        name = f'<a href="{name}" target="_blank" rel="noopener noreferrer">{name}</a>'
    page = f' <i>page {s["page"]}</i>' if s["page"] else ""
    return (f'<div class="x-src"><div class="h">[{i}] {name}{page}</div>'
            f'<div class="t">{html.escape(s["text"])}</div></div>')


def render_extras(msg):
    """Timing line plus one collapsed panel: sources, agent steps, copyable markdown."""
    sources, trace = msg.get("sources", []), msg.get("trace", [])
    if msg.get("failed") or not (sources or trace):
        return
    st.markdown(
        f'<div class="x-meta">Answered in {msg.get("elapsed", 0):.1f} s from '
        f'{plural(len(sources), "source")} in {plural(len(trace), "step")}.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Sources and steps"):
        tab_src, tab_steps, tab_raw = st.tabs(
            [f"Sources ({len(sources)})", f"Steps ({len(trace)})", "Copy as markdown"]
        )
        with tab_src:
            if sources:
                st.markdown("".join(source_html(i, s) for i, s in enumerate(sources, 1)),
                            unsafe_allow_html=True)
            else:
                st.caption("No documents were used for this answer.")
        with tab_steps:
            st.markdown(timeline_html(trace), unsafe_allow_html=True)
        with tab_raw:
            show_code(msg["content"], "markdown")


def show_message(msg):
    with st.chat_message(msg["role"]):
        render_answer(msg["content"])
        if msg["role"] == "assistant":
            render_extras(msg)


def chat_markdown():
    return "\n\n---\n\n".join(
        f"**{'You' if m['role'] == 'user' else 'Agent'}**\n\n{m['content']}"
        for m in st.session_state.messages
    )


# --------------------------------------------------------------------- layout
def upload_sidebar():
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    with st.sidebar:
        st.subheader("Documents")

        # Message saved before st.rerun() so it survives the rerun
        if "upload_msg" in st.session_state:
            st.success(st.session_state.pop("upload_msg"))

        files = st.file_uploader(
            "Upload files",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )

        if st.button("Add to knowledge base", disabled=not files, type="primary"):
            data_dir = Path(config.DATA_DIR)
            data_dir.mkdir(parents=True, exist_ok=True)

            for f in files:
                # Path(...).name strips any directory parts from the filename
                (data_dir / Path(f.name).name).write_bytes(f.getbuffer())

            try:
                with st.spinner("Indexing documents..."):
                    chunks = load_and_split_documents(config.DATA_DIR)
                    build_vectorstore(chunks, persist_directory=config.CHROMA_DIR)
            except Exception as e:
                st.error(f"Indexing failed: {e}")
            else:
                get_retriever.cache_clear()  # drop the stale BM25 index
                get_graph.clear()            # rebuild the graph with the new retriever
                get_graph_png.clear()
                st.session_state.upload_msg = (
                    f"Indexed {len(chunks)} chunks from {len(files)} new file(s)."
                )
                st.session_state.uploader_key += 1  # reset the uploader
                st.rerun()

        indexed = list_indexed_files()
        if indexed:
            st.caption(f"Indexed files ({len(indexed)})")
            st.markdown(
                '<div class="x-files">' + "".join(f"<div>{html.escape(n)}</div>" for n in indexed) + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No documents yet. Upload some above.")

        st.divider()
        st.download_button(
            "Export chat as Markdown", chat_markdown(), "chat.md", "text/markdown",
            disabled=not st.session_state.messages,
        )
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        with st.expander("Agent graph"):
            # Rendered on demand: the PNG comes from mermaid.ink over the network
            if st.button("Show diagram"):
                png = get_graph_png()
                if png:
                    st.image(png)
                else:
                    st.caption("Diagram unavailable (needs internet access or a working graph).")


def empty_state():
    cols = st.columns(2)
    for i, text in enumerate(EXAMPLES):
        if cols[i % 2].button(text, key=f"example_{i}"):
            st.session_state.pending = text
            st.rerun()


def main():
    st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("messages", [])

    # Sidebar first, so you can upload documents even if setup fails
    # (for example, on the first launch when DATA_DIR is still empty).
    upload_sidebar()

    st.markdown(
        '<div class="x-title">Agentic RAG</div>'
        '<div class="x-lede">Ask about your documents. The agent routes, retrieves and grades before it answers.</div>',
        unsafe_allow_html=True,
    )

    try:
        graph = get_graph()
    except Exception as e:
        st.error(f"Setup failed: {e}")
        st.info("If no documents are indexed yet, upload some in the sidebar.")
        st.stop()

    question = st.chat_input("Ask a question about your documents") or st.session_state.pop("pending", None)

    if not st.session_state.messages and not question:
        empty_state()
    for msg in st.session_state.messages:
        show_message(msg)

    if not question:
        return

    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    show_message(user_msg)

    with st.chat_message("assistant"):
        answer, docs, trace = "", [], []
        err_msg = err_trace = None
        t0 = time.perf_counter()
        with st.status("Agent is working...", expanded=True) as status:
            timeline = st.empty()
            try:
                for node_name, update in run_agent(graph, question):
                    trace.append(describe_step(node_name, update, question))
                    timeline.markdown(timeline_html(trace), unsafe_allow_html=True)
                    if "documents" in update:
                        docs = update["documents"] or []
                    if update.get("generation"):
                        answer = update["generation"]
                status.update(
                    label=f"Finished in {time.perf_counter() - t0:.1f} s",
                    state="complete", expanded=False,
                )
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                err_trace = traceback.format_exc()
                status.update(label="Error", state="error", expanded=True)

        msg = {"role": "assistant", "trace": trace, "elapsed": time.perf_counter() - t0}
        if err_msg:
            st.error(err_msg)
            with st.expander("Full traceback"):
                st.code(err_trace)
            msg.update(content=f"⚠️ The agent failed: {err_msg}", sources=[], failed=True)
        else:
            msg.update(content=answer or "(no answer generated)", sources=[to_source(d) for d in docs])
            render_answer(msg["content"])
            render_extras(msg)

    st.session_state.messages.append(msg)


if __name__ == "__main__":
    main()