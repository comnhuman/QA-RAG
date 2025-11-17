import logging
from src.state import State
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# LLM_ID = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
LLM_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
# LLM_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
# LLM_ID = "Qwen/Qwen3-8B"
OPENAI_URL = "http://127.0.0.1:9805/v1"

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary scoring to evaluate the appropriateness of answers to questions"""

    binary_score: str = Field(
        description="Indicate 'yes' or 'no' whether the answer solves the question"
    )


try:
    logger.info(
        f"Initializing hallucination checker LLM: model='{LLM_ID}', url='{OPENAI_URL}'"
    )
    llm = ChatOpenAI(
        api_key="EMPTY",
        base_url=OPENAI_URL,
        model=LLM_ID,
    )
    structured_h_llm_grader = llm.with_structured_output(GradeHallucinations)
    structured_a_llm_grader = llm.with_structured_output(GradeAnswer)
    logger.info("Hallucination checker LLM initialized successfully")
except Exception as e:
    logger.exception(f"Failed to initialize hallucination checker LLM: {e}")
    raise

# 프롬프트 설정
system_h = """You are a grader assessing whether an LLM generation is grounded in and supported by a set of retrieved facts.

Your goal is to detect hallucinations: concrete factual statements that are not supported by the provided facts.

Scoring rules:
- Return "yes" if:
  - All factual statements in the generation can be directly found in the provided facts, or can be reasonably inferred from them; OR
  - The generation explicitly states that it cannot answer the question or that the context is insufficient, and it does not introduce any new concrete factual details (such as specific numbers, dates, names, locations, or mechanisms) that are not present in the facts.
- Return "no" if:
  - The generation contains concrete factual claims (for example specific numbers, dates, locations, causal explanations, or detailed statistics) that are not supported by the provided facts; OR
  - The generation clearly contradicts the facts.

Clarifications:
- "Reasonably inferred" means paraphrasing, summarizing, or combining information that is already in the facts without adding new entities, new numeric values, new dates, or new specific causal explanations that are not mentioned.
- Ignore purely structural or formatting text such as headings (like "Source"), template placeholders, or list markers; only evaluate the semantic factual content of the generation.

You must choose a single value for the binary_score field: "yes" if the answer is grounded as described above, otherwise "no"."""

system_a = """You are a grader assessing whether an answer resolves a user question in the context of a Retrieval-Augmented Generation (RAG) system.

Your task is to read the user question and the LLM's answer, then decide whether the answer can be considered as resolving the question for the purposes of this QA system.

Scoring rules:
- Return "yes" if:
  - The answer directly and sufficiently addresses the main requirements of the question; OR
  - The answer clearly and explicitly states that it cannot answer the question based on the available information or context (for example, it says there is no relevant data or not enough information). In a RAG system, such an explicit admission of insufficient information is also considered a valid resolution.
- Return "no" if:
  - The answer is off-topic, ignores key parts of the question, is mostly irrelevant, or is clearly incomplete or nonsensical; OR
  - The answer pretends to know the answer while providing obviously invented, hallucinated, or unsupported details instead of acknowledging missing information.

Clarifications:
- Focus on whether the answer meaningfully addresses the question or explicitly explains why it cannot do so; do not grade fine-grained factual correctness here, as that is handled by a separate hallucination checker.

You must choose a single value for the binary_score field: "yes" if the answer resolves the question as described above, otherwise "no"."""


# 프롬프트 템플릿 생성
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_h),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_a),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

# 환각 평가기 생성
hallucination_grader = hallucination_prompt | structured_h_llm_grader
answer_grader = answer_prompt | structured_a_llm_grader



def hallucination_check(state: State):
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    if not generation:
        logger.warning("hallucination_check() called with empty generation.")
        return

    docs_len = len(documents) if documents is not None else 0
    logger.info(
        f"Running hallucination check for question: '{question}' (docs={docs_len})"
    )

    try:
        # 환각 평가
        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        h_grade = score.binary_score
        logger.info(f"Hallucination grade result: {h_grade}")

        # Hallucination 여부 확인
        if h_grade == "yes":
            # 답변의 관련성(Relevance) 평가
            logger.info(
                "Generation is grounded in documents. Running relevance check against question."
            )
            score = answer_grader.invoke({"question": question, "generation": generation})
            a_grade = score.binary_score
            logger.info(f"Answer relevance grade result: {a_grade}")

            # 관련성 평가 결과에 따른 처리
            if a_grade == "yes":
                logger.info("Generation is relevant: it addresses the user question.")
                return "relevant"
            else:
                logger.info(
                    "Generation is grounded but not sufficiently relevant to the question."
                )
                return "not relevant"
        else:
            logger.warning(
                "Generation is not grounded in documents. Marking as hallucination."
            )
            return "hallucination"

    except Exception as e:
        logger.exception(
            f"Error during hallucination_check for question='{question}': {e}"
        )
        raise