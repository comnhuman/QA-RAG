import os
import logging
from pathlib import Path

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "embedding_pipeline_test.log"

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger("embedding_pipeline")

os.makedirs("./huggingface_data", exist_ok=True)
os.environ["HF_HOME"] = "./huggingface_data"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import hashlib
import time
from typing import Any, Iterable
from openai import OpenAI
from pymilvus import MilvusClient, DataType
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from langchain_core.documents import Document


DATA_DIR = Path("data")
EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
# EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-4B"
OPENAI_URL = "http://127.0.0.1:9804/v1"
MILVUS_URI = "http://127.0.0.1:19530"
EXPORT_TYPE = "doc_chunks"
CHUNKER = HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=1000)

logger.info("클라이언트 초기화(OpenAI/Milvus)")
openai_client = OpenAI(
    api_key="EMPTY",
    base_url=OPENAI_URL,
)

milvus_client = MilvusClient(
    uri=MILVUS_URI,
    token="root:Milvus"
)
logger.info("OpenAI/Milvus 클라이언트 생성 완료")

logger.info("Milvus 데이터베이스/컬렉션 준비")
db_name = "doc_embeddings2"
existing_dbs = milvus_client.list_databases()
if db_name not in existing_dbs:
    logger.info("DB 생성: %s", db_name)
    milvus_client.create_database(db_name=db_name)
else:
    logger.info("DB 이미 존재: %s", db_name)
milvus_client.use_database(db_name)

collection_name = "doc_embeddings"

all_files = [
    f for f in DATA_DIR.rglob("*")
    if f.is_file() and not f.name.startswith(".")
]

logger.info(f"발견된 파일 수: {len(all_files)}")

all_docs = []
for file_path in all_files:
    try:
        loader = DoclingLoader(
            file_path=str(file_path),
            export_type=EXPORT_TYPE,
            chunker=CHUNKER,
        )
        docs = loader.load()
        all_docs.extend(docs)
        logger.info(f"{file_path.name} → {len(docs)} chunks 생성 완료")
    except Exception as e:
        logger.exception(f"{file_path.name} 처리 중 오류 발생: {e}")

logger.info(f"총 청크 수: {len(all_docs)}")

from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus

embeddings = OpenAIEmbeddings(
    api_key="EMPTY",
    base_url=OPENAI_URL,
    model=EMBED_MODEL_ID
)

db_name="doc_embeddings2"
existing_dbs = milvus_client.list_databases()

if db_name not in existing_dbs:
    milvus_client.create_database(db_name=db_name)

vectorstore = Milvus.from_documents(
    documents=all_docs,
    embedding=embeddings,
    collection_name="doc_embeddings",
    connection_args={"uri": MILVUS_URI, "token": "root:Milvus", "db_name": db_name},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    drop_old=False,
)

logger.info(f"\n완료")