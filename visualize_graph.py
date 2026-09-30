"""
Export the agentic RAG graph as a Mermaid source file and a PNG image.

Usage:
    python draw_graph.py
"""
from src import config
from src.graph import build_graph

MERMAID_PATH = "graph_diagram.mmd"
PNG_PATH = "graph_diagram.png"


def main():
    config.check_required_api_keys()
    app = build_graph()
    graph = app.get_graph()

    mermaid_text = graph.draw_mermaid()
    with open(MERMAID_PATH, "w", encoding="utf-8") as f:
        f.write(mermaid_text)
    print(f"Wrote Mermaid source to {MERMAID_PATH}")
    print("\n" + mermaid_text)

    try:
        png_bytes = graph.draw_mermaid_png(output_file_path=PNG_PATH)
        print(f"Wrote PNG diagram to {PNG_PATH} ({len(png_bytes)} bytes)")
    except Exception as exc:
        print(
            f"Could not render PNG ({exc}). The Mermaid source above still "
            "works: paste it into https://mermaid.live to view/export it."
        )


if __name__ == "__main__":
    main()