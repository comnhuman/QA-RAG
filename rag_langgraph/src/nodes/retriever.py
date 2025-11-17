import logging, time, threading
from src.state import State
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.messages import HumanMessage
from langgraph.graph import END

logger = logging.getLogger(__name__)

EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
# EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-4B"
OPENAI_URL = "http://127.0.0.1:9804/v1"
MILVUS_URI = "http://127.0.0.1:19530"
TIMEOUT_SEC = 30

def test_embedding_connection(embeddings: OpenAIEmbeddings, test_text: str, timeout: int) -> None:
    result_container = {"result": None, "error": None}

    def run():
        try:
            result_container["result"] = embeddings.embed_query(test_text)
        except Exception as e:
            result_container["error"] = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"Embedding call exceeded timeout of {timeout} seconds.")

    if result_container["error"]:
        raise result_container["error"]

    result = result_container["result"]
    if result and isinstance(result, list) and len(result) > 0:
        logger.info(f"Embedding model connected successfully")
    else:
        raise ValueError("Embedding response is invalid or empty.")

try:
    logger.info(f"Initializing Embedding model: model='{EMBED_MODEL_ID}', url='{OPENAI_URL}'")
    embeddings = OpenAIEmbeddings(
        api_key="EMPTY",
        base_url=OPENAI_URL,
        model=EMBED_MODEL_ID,
        tiktoken_enabled=False
    )

    test_embedding_connection(embeddings, "연결 테스트 문장", TIMEOUT_SEC)
    
except Exception as e:
    logger.exception(f"Failed to initialize OpenAIEmbeddings: {e}")
    raise

try:
    logger.info(f"Connecting to Milvus at {MILVUS_URI}")
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name="doc_embeddings",
        connection_args={
            "uri": MILVUS_URI,
            "token": "root:Milvus",
            "db_name": "doc_embeddings"
        },
        index_params={
            "index_type": "FLAT",
            "metric_type": "L2"
        },
    )
    logger.info("Milvus vector store connection established.")

except Exception as e:
    logger.exception(f"Failed to connect to Milvus: {e}")
    raise

def retrieve(state: State):
    question = state.get("question", "").strip()
    human_message = HumanMessage(question)

    if not question:
        logger.warning("retrieve() called with empty question in state.")
        return END

    logger.info(f"Retrieving documents for question: '{question}'")

    try:
        documents = vector_store.similarity_search(question, k=20)
        logger.info(f"Retrieved {len(documents)} documents from vector store.")
        return {"documents": documents, "question": question, "messages": [human_message]}

    except Exception as e:
        logger.exception(f"Error during retrieval for question='{question}': {e}")
        return END
