import logging
import time
from src.state import State
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# LLM_ID = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
LLM_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
# LLM_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
# LLM_ID = "Qwen/Qwen3-8B"
OPENAI_URL = "http://127.0.0.1:9805/v1"

# Query Rewriter용 LLM 초기화
try:
    logger.info(
        f"Initializing query rewriter LLM: model='{LLM_ID}', url='{OPENAI_URL}'"
    )
    rewriter_llm = ChatOpenAI(
        api_key="EMPTY",
        base_url=OPENAI_URL,
        model=LLM_ID,
    )
    logger.info("Query rewriter LLM initialized successfully")
except Exception as e:
    logger.exception(f"Failed to initialize query rewriter LLM: {e}")
    raise

# Query Rewriter 프롬프트 정의
system = """You are a question re-writer that converts an input question into a better version
that is optimized for vectorstore retrieval.
Focus on clarifying the underlying semantic intent and important keywords,
while preserving the original meaning of the question.

Rules:
- Keep the OUTPUT in the same language as the input question.
- Do NOT translate the question unless explicitly asked.
- Make the query concise and keyword-focused (no long sentences).
- Preserve all important entities (names, locations, dates, numbers, constraints).
- Do NOT add explanations, comments, or bullet points.
- Output ONLY the rewritten search query, nothing else.
"""

# Query Rewriter 프롬프트 템플릿 생성
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question:\n\n{question}\n\nRewrite this as an improved search query.",
        ),
    ]
)


# Query Rewriter 체인 생성
question_rewriter_chain = re_write_prompt | rewriter_llm | StrOutputParser()


def transform_query(state: State):
    """State에 들어있는 question을 벡터검색에 더 적합한 형태로 재작성한다."""
    question = state["question"]

    if not question:
        logger.warning("rewrite_question() called with empty question in state.")
        return {"question": ""}

    logger.info(f"Rewriting question: '{question}'")

    try:
        start_time = time.time()
        rewritten = question_rewriter_chain.invoke({"question": question})
        elapsed = time.time() - start_time

        # 앞뒤 공백 정리
        rewritten = rewritten.strip()
        logger.info(
            f"Question rewriting complete in {elapsed:.2f}s "
            f"(rewritten='{rewritten}')"
        )

        # 필요에 따라 원문을 따로 저장하고 싶으면 여기서 추가로 반환 가능
        # 예: return {"question": rewritten, "original_question": question}
        return {"question": rewritten}

    except Exception as e:
        logger.exception(
            f"Error during question rewriting for question='{question}': {e}"
        )
        # 문제가 생기면 안전하게 원래 질문을 그대로 사용
        return {"question": question}
