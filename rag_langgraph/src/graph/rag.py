import logging
from src.state import State
from src.nodes.retriever import retrieve
from src.nodes.generator import generate
from src.nodes.reranker import rerank
from src.nodes.hallucination_checker import hallucination_check
from src.nodes.query_rewriter import transform_query
from langgraph.graph import END, StateGraph, START

logger = logging.getLogger(__name__)

try:
    logger.info("Initializing RAG workflow graph")
    workflow = StateGraph(State)

    logger.info("Adding nodes")
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    logger.info("Defining workflow edges")
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "rerank") 
    workflow.add_edge("rerank", "generate")

    workflow.add_conditional_edges(
        "generate",
        hallucination_check,
        {
            "hallucination": "generate",  # Hallucination 발생 시 재생성
            "relevant": END,  # 답변의 관련성 여부 통과
            "not relevant": "transform_query",  # 답변의 관련성 여부 통과 실패 시 쿼리 변환
        },
    )
    
    workflow.add_edge("transform_query", "retrieve")

    rag = workflow.compile()
    logger.info("RAG workflow compiled successfully.")

except Exception as e:
    logger.exception(f"Failed to build RAG workflow: {e}")
    raise

# try:
#     logger.info("Initializing RAG workflow graph")
#     workflow = StateGraph(State)

#     logger.info("Adding nodes")
#     workflow.add_node("retrieve", retrieve)
#     workflow.add_node("rerank", rerank)
#     # workflow.add_node("generate", generate)

#     logger.info("Defining workflow edges")
#     workflow.add_edge(START, "retrieve")
#     workflow.add_edge("retrieve", "rerank") 
#     # workflow.add_edge("rerank", "generate")
#     workflow.add_edge("rerank", END)

#     rag = workflow.compile()
#     logger.info("RAG workflow compiled successfully.")

# except Exception as e:
#     logger.exception(f"Failed to build RAG workflow: {e}")
#     raise