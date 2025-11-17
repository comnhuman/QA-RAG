import logging

import httpx
from langchain_core.documents import Document
from langgraph.graph import END
from src.state import State

logger = logging.getLogger(__name__)

RERANK_API_URL = "http://127.0.0.1:9806/rerank"
RERANK_MODEL = "Qwen/Qwen3-Reranker-8B"
# RERANK_MODEL = "Qwen/Qwen3-Reranker-4B"

RERANK_TOP_K = 5
RERANK_MIN_SCORE = 0.5
RERANK_TIMEOUT = 300

def _to_texts_and_keep(documents: list[Document]) -> tuple[list[str], list[Document]]:
    texts: list[str] = []
    kept: list[Document] = []

    for doc in documents:
        if hasattr(doc, "page_content"):
            text = getattr(doc, "page_content", None)
        else:
            text = str(doc)

        texts.append(text if text is not None else "")
        kept.append(doc)

    return texts, kept


def _build_payload(question: str, texts: list[str]) -> dict:
    payload: dict = {
        "model": RERANK_MODEL,
        "query": question,
        "documents": texts,
        "top_n": RERANK_TOP_K
    }

    return payload


def _post_rerank(question: str, texts: list[str]) -> tuple[list[int], dict[int, float]]:
    payload = _build_payload(question, texts)

    logger.info(f"Calling Rerank API: url={RERANK_API_URL}, n_docs={len(texts)}")
    with httpx.Client(timeout=RERANK_TIMEOUT) as client:
        resp = client.post(RERANK_API_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Invalid rerank response (results not list): {data}")

    order = []
    scores_map = {}
    for r in results:
        idx = r.get("index", None)
        score = r.get("relevance_score", None)

        if idx is None or score is None:
            logger.warning(f"Skipping invalid result item: {r}")
            continue
        
        try:
            i = int(idx)
            s = float(score)
        except Exception:
            logger.warning(f"Skipping unparsable result item: {r}")
            continue

        if i < 0 or i >= len(texts):
            logger.warning(f"Result index out of range: {i} (n_docs={len(texts)})")
            continue
        order.append(i)
        scores_map[i] = s

    return order, scores_map


def _apply_topk_and_threshold(order: list[int], scores_map: dict[int, float]) -> list[int]:
    try:
        thr = float(RERANK_MIN_SCORE)
        order = [i for i in order if scores_map.get(i, float("-inf")) >= thr]
    except Exception:
        logger.warning("RERANK_MIN_SCORE is not a float; ignoring threshold filter.")

    return order


def rerank(state: State):
    question = state.get("question")
    documents = state.get("documents")
    # messages = state.get("messages", [])  # 현재 노드에서는 사용하지 않음

    if not documents:
        logger.info("rerank() no documents to rerank; returning state unchanged.")
        return END

    try:
        texts, kept = _to_texts_and_keep(documents)

        order, scores_map = _post_rerank(question, texts)
        if not order:
            logger.info("rerank() got empty results from reranker; returning state unchanged.")
            return

        order = _apply_topk_and_threshold(order, scores_map)
        reranked_docs = [kept[i] for i in order]

        preview_n = min(5, len(reranked_docs))
        preview_scores = [round(scores_map[i], 4) for i in order[:preview_n]]
        logger.info(
            f"Reranked {len(kept)} docs. "
            f"Selected {len(reranked_docs)} after filters. "
            f"Top-{preview_n} scores: {preview_scores}"
        )
        return {"documents": reranked_docs}

    except Exception as e:
        logger.exception(f"Failed to rerank documents: {e}")
        return END
