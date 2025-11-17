import os
import logging
from pathlib import Path

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "embedding_pipeline_test2.log"

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
CHUNKER = HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=500)

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
db_name = "doc_embeddings3"
existing_dbs = milvus_client.list_databases()
if db_name not in existing_dbs:
    logger.info("DB 생성: %s", db_name)
    milvus_client.create_database(db_name=db_name)
else:
    logger.info("DB 이미 존재: %s", db_name)
milvus_client.use_database(db_name)

collection_name = "doc_embeddings"
if not milvus_client.has_collection(collection_name):
    logger.info("컬렉션 생성: %s", collection_name)
    schema = milvus_client.create_schema()
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535,
    )
    schema.add_field(
        field_name="pk",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=False,
    )
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=4096,
        # dim=2560,
    )
    schema.add_field(
        field_name="source",
        datatype=DataType.JSON,
    )
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_name = "vector",
        index_type="FLAT",
        metric_type="COSINE"
    )
    milvus_client.create_collection(collection_name=collection_name, index_params=index_params, schema=schema)
else:
    logger.info("컬렉션 이미 존재: %s", collection_name)

def dedup_convert_inplace(docs: list[Document]) -> None:
    logger.debug("중복제거 시작 (입력 문서 수=%d)", len(docs))
    seen_index: dict[int, int] = {} 
    remove_indices: list[int] = []

    for i in range(len(docs)):
        doc = docs[i]
        md = getattr(doc, "metadata", {}) or {}

        src = md.get("source")
        if isinstance(src, list):
            src_list = [s for s in src if s is not None]
        elif src is None:
            src_list = []
        else:
            src_list = [src]

        pk = md.get("pk")
        if pk is None:
            hash_val = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
            pk = int(hash_val, 16) % (1 << 63)

        if pk not in seen_index:
            docs[i] = {
                "text": doc.page_content,
                "pk": pk,
                "source": src_list,
            }
            seen_index[pk] = i
        else:
            base = docs[seen_index[pk]]       
            base_src = base.get("source", [])
            for s in src_list:
                if s not in base_src:
                    base_src.append(s)
            base["source"] = base_src

            remove_indices.append(i)

    for idx in reversed(remove_indices):
        del docs[idx]
    logger.info("중복제거 완료")

def _chunks(seq: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def embed_docs_inplace(
    docs: list[dict[str, Any]],
    *,
    model: str = EMBED_MODEL_ID,
    batch_size: int = 128,
    text_key: str = "text",
    vector_key: str = "vector",
    max_retries: int = 5,
    backoff_base: float = 1.5,
) -> None:
    logger.info("임베딩 시작")
    for batch in _chunks(docs, batch_size):
        to_process = [(i, d) for i, d in enumerate(batch)]

        if not to_process:
            continue

        inputs = []
        idx_map = []
        for j, d in to_process:
            text = d.get(text_key)
            if not isinstance(text, str) or not text.strip():
                d[vector_key] = None
                continue
            inputs.append(text)
            idx_map.append(j)

        if not inputs:
            continue

        attempt = 0
        while True:
            try:
                resp = openai_client.embeddings.create(model=model, input=inputs)
                for k, item in enumerate(resp.data):
                    batch[idx_map[k]][vector_key] = item.embedding
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    for j in idx_map:
                        batch[j][vector_key] = None
                    break
                time.sleep(backoff_base ** attempt)
    logger.info("임베딩 종료")


def fetch_existing_sources(
    client: MilvusClient, collection: str, pks: list[int], *, batch: int = 1000
) -> dict[int, list]:
    existing: dict[int, list] = {}
    for chunk in _chunks(pks, batch):
        if not chunk:
            continue
        # INT64 PK 기준 필터
        filter_expr = f"pk in [{','.join(map(str, chunk))}]"
        rows = client.query(
            collection_name=collection,
            filter=filter_expr,
            output_fields=["pk", "source"],
        ) or []
        for r in rows:
            s = r.get("source") or []
            if isinstance(s, str):
                try:
                    s = json.loads(s)
                except Exception:
                    s = [s]
            existing[int(r["pk"])] = s
    return existing

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

dedup_convert_inplace(all_docs)
embed_docs_inplace(all_docs)

logger.info("Milvus 삽입")
all_pks = [d["pk"] for d in all_docs if "pk" in d]
exist_map = fetch_existing_sources(milvus_client, collection_name, all_pks)

for d in all_docs:
    cur = d.get("source") or []
    prev = exist_map.get(d["pk"]) or []
    d["source"] = list(set(prev + cur))

existing_pks = set(exist_map.keys())
to_update_minimal = [
    {"pk": d["pk"], "source": d["source"]}  
    for d in all_docs if d["pk"] in existing_pks
]
to_insert_full = [
    d for d in all_docs if d["pk"] not in existing_pks 
]

if to_update_minimal:
    try:
        milvus_client.upsert(
            collection_name=collection_name,
            data=to_update_minimal,
            partial_update=True, 
        )
        logger.info("기존 PK %d건 source 부분 업데이트 완료", len(to_update_minimal))
    except Exception as e:
        logger.error("partial_update 실패: %s", e)

if to_insert_full:
    milvus_client.upsert(collection_name=collection_name, data=to_insert_full)
    logger.info("신규 PK %d건 upsert 완료", len(to_insert_full))

logger.info(f"완료")


# from langchain_openai import OpenAIEmbeddings
# from langchain_milvus import Milvus

# embeddings = OpenAIEmbeddings(
#     api_key="EMPTY",
#     base_url=OPENAI_URL,
#     model=EMBED_MODEL_ID
# )

# db_name="doc_embeddings"
# existing_dbs = milvus_client.list_databases()

# if db_name not in existing_dbs:
#     milvus_client.create_database(db_name=db_name)

# vectorstore = Milvus.from_documents(
#     documents=all_docs,
#     embedding=embeddings,
#     collection_name="doc_embeddings",
#     connection_args={"uri": MILVUS_URI, "token": "root:Milvus", "db_name": db_name},
#     index_params={"index_type": "FLAT", "metric_type": "L2"},
#     drop_old=False,
# )

# logger.info(f"\n완료")