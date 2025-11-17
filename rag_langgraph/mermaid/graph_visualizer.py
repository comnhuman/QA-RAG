import os
from pathlib import Path

os.makedirs("../huggingface_data", exist_ok=True)
os.environ["HF_HOME"] = "../huggingface_data"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.graph.rag import rag
app = rag

png_bytes = app.get_graph().draw_mermaid_png()

path = Path("mermaid/rag_graph_v4.png")
with open(path, "wb") as f:
    f.write(png_bytes)