import os
import logging
from pathlib import Path

os.makedirs("../huggingface_data", exist_ok=True)
os.environ["HF_HOME"] = "../huggingface_data"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "rag_pipeline.log"

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)

from src.graph.rag import rag
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag_pipeline")

q = "튀니지정부와 수행한 사업에 대해 알려줘"
result = rag.invoke({"question": q})["generation"]

print()
print(result)