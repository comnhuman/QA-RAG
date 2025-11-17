import os
import logging
from pathlib import Path

os.makedirs("../huggingface_data", exist_ok=True)
os.environ["HF_HOME"] = "../huggingface_data"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "embedding_pipeline.log"

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger("embedding_pipeline")

import json
import hashlib
import time
from typing import Any, Iterable
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_text_splitters import TokenTextSplitter
from openai import OpenAI
from pymilvus import MilvusClient, DataType
from docling.chunking import HybridChunker
from custom_loader import HWPLoader
from transformers import AutoTokenizer

DATA_DIR = Path("data")
DATA_DIR = Path("sample_data")
EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
# EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-4B"
OPENAI_URL = "http://127.0.0.1:9804/v1"
MILVUS_URI = "http://127.0.0.1:19530"
EXPORT_TYPE = "doc_chunks"
CHUNKER = HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=1000)
TEXT_SPLITTER = TokenTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    chunk_size=1000,
    chunk_overlap = 10
)

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
db_name = "doc_embeddings"
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
        metric_type="L2"
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

DOCLING_SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".md", ".markdown",
    ".adoc", ".asciidoc",
    ".html", ".htm", ".xhtml",
    ".csv",
    ".png",
    ".jpg", ".jpeg",
    ".tif", ".tiff",
    ".bmp",
    ".webp",
    ".vtt",   # WebVTT
}

BATCH_FILE_SIZE = 10  # all_files를 10개씩 처리

unsupported_files = []
total_chunks = 0

# _chunks는 위에서 정의된 함수를 그대로 사용합니다.
for file_batch in _chunks(all_files, BATCH_FILE_SIZE):
    logger.info("파일 배치 처리 시작 (배치 크기=%d)", len(file_batch))

    batch_docs = []

    # 1) 파일 로드 + 청크 생성
    for file_path in file_batch:
        suffix = file_path.suffix.lower()

        if suffix != ".hwp" and suffix not in DOCLING_SUPPORTED_SUFFIXES:
            unsupported_files.append(file_path)
            continue

        try:
            if suffix == ".hwp":
                loader = HWPLoader(
                    file_path=str(file_path)
                )
                docs = loader.load()
                docs = TEXT_SPLITTER.split_documents(docs)
            else:
                loader = DoclingLoader(
                    file_path=str(file_path),
                    export_type=EXPORT_TYPE,
                    chunker=CHUNKER,
                )
                docs = loader.load()

            batch_docs.extend(docs)
            logger.info("%s → %d chunks 생성 완료", file_path.name, len(docs))

        except Exception as e:
            logger.exception("%s 처리 중 오류 발생: %s", file_path.name, e)

    if not batch_docs:
        logger.info("현재 배치에서 생성된 청크가 없습니다. 다음 배치로 넘어갑니다.")
        continue

    total_chunks += len(batch_docs)
    logger.info("현재 배치 청크 수: %d (누적 청크 수: %d)", len(batch_docs), total_chunks)

    # 2) 중복 제거 + dict 형태로 변환
    dedup_convert_inplace(batch_docs)

    # 3) 임베딩
    embed_docs_inplace(batch_docs)

    # 4) Milvus 삽입/업데이트 (현재 배치만)
    logger.info("Milvus 삽입 (현재 배치)")

    all_pks = [d["pk"] for d in batch_docs if "pk" in d]
    exist_map = fetch_existing_sources(milvus_client, collection_name, all_pks)

    for d in batch_docs:
        cur = d.get("source") or []
        prev = exist_map.get(d["pk"]) or []
        d["source"] = list(set(prev + cur))

    existing_pks = set(exist_map.keys())
    to_update_minimal = [
        {"pk": d["pk"], "source": d["source"]}
        for d in batch_docs if d["pk"] in existing_pks
    ]
    to_insert_full = [
        d for d in batch_docs if d["pk"] not in existing_pks
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

    # 배치 완료 후 메모리 해제 도움
    del batch_docs

# 전체 파일 순회가 끝난 후 unsupported 파일 로그
if unsupported_files:
    skipped_names = ", ".join(f.name for f in unsupported_files)
    logger.warning(
        "어떤 로더에도 매핑되지 않아 스킵된 파일 %d개: %s",
        len(unsupported_files),
        skipped_names,
    )

logger.info("전체 처리 완료 (총 청크 수: %d)", total_chunks)