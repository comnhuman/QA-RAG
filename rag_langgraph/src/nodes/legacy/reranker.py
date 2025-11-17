import logging

import httpx
from langchain_core.documents import Document
from src.state import State

logger = logging.getLogger(__name__)

RERANK_API_URL = "http://127.0.0.1:9806/rerank"
RERANK_TIMEOUT_SEC = 15
RERANK_TOP_K = 5
RERANK_MIN_SCORE = 0.5
RERANK_INSTRUCTION = "Given a user query, retrieve relevant passages from the documents that answer the query."

def _to_texts_and_keep(documents: list[Document]) -> tuple[list[str], list[Document]]:
    texts: list[str] = []
    kept: list[Document] = []
    for doc in documents:
        text = None
        if hasattr(doc, "page_content"):
            text = getattr(doc, "page_content", None)
        else:
            text = str(doc)

        texts.append(text)
        kept.append(doc)
    return texts, kept


def _post_rerank(question: str, texts: list[str]) -> list[float]:
    payload = {
        "queries": [question] * len(texts),
        "documents": texts,
        "instruction": RERANK_INSTRUCTION,
    }

    logger.info(f"Calling Rerank API: url={RERANK_API_URL}, n_docs={len(texts)}")
    # with httpx.Client(timeout=httpx.Timeout(RERANK_TIMEOUT_SEC)) as client:
    with httpx.Client(timeout=None) as client:
        resp = client.post(RERANK_API_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    scores = data.get("scores", [])
    if not isinstance(scores, list) or len(scores) != len(texts):
        raise ValueError(f"Invalid rerank response: {data}")

    return [float(s) for s in scores]


def _apply_topk_and_threshold(order: list[int], scores: list[float]) -> list[int]:
    if RERANK_MIN_SCORE is not None:
        try:
            thr = float(RERANK_MIN_SCORE)
            order = [i for i in order if scores[i] >= thr]
        except Exception:
            logger.warning("RERANK_MIN_SCORE is not a float; ignoring threshold filter.")

    # TOP_K 제한
    if RERANK_TOP_K is not None:
        try:
            k = RERANK_TOP_K
            if k > 0:
                order = order[:k]
        except Exception:
            logger.warning("RERANK_TOP_K is not an int; ignoring top-k.")

    return order


def rerank(state: State):
    question = state.get("question")
    documents = state.get("documents")
    messages = state.get("messages", [])

    if not question:
        logger.warning("rerank() called with empty question; returning state unchanged.")
        return

    if not documents:
        logger.info("rerank() no documents to rerank; returning state unchanged.")
        return

    try:
        texts, kept = _to_texts_and_keep(documents)
        scores = _post_rerank(question, texts)

        order = sorted(range(len(kept)), key=lambda i: scores[i], reverse=True)
        order = _apply_topk_and_threshold(order, scores)

        reranked_docs = [kept[i] for i in order]

        # 로그: 상위 몇 개만 미리보기
        preview_n = min(5, len(reranked_docs))
        logger.info(
            f"Reranked {len(kept)} docs. "
            f"Top-{preview_n} scores: {[round(scores[i], 4) for i in order[:preview_n]]}"
        )
        return {"documents": reranked_docs}

    except Exception as e:
        logger.exception(f"Failed to rerank documents: {e}")
        return
