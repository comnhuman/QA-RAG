import logging, time, threading
from src.state import State
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.messages import AIMessage
from langgraph.graph import END

logger = logging.getLogger(__name__)

# LLM_ID = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
LLM_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
# LLM_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
# LLM_ID = "Qwen/Qwen3-8B"
OPENAI_URL = "http://127.0.0.1:9805/v1"
TIMEOUT_SEC = 15

def test_llm_connection(llm: ChatOpenAI, prompt: str, timeout: int) -> None:
    result_container = {"result": None, "error": None}

    def run():
        try:
            result_container["result"] = llm.invoke(prompt)
        except Exception as e:
            result_container["error"] = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"LLM call exceeded timeout of {timeout} seconds.")

    if result_container["error"]:
        raise result_container["error"]

    result = result_container["result"]
    if hasattr(result, "content") and result.content:
        logger.info(f"LLM connected successfully")
    else:
        raise ValueError("LLM response is invalid or empty.")

try:
    logger.info(f"Initializing LLM: model='{LLM_ID}', url='{OPENAI_URL}'")
    llm = ChatOpenAI(
        api_key="EMPTY",
        base_url=OPENAI_URL,
        model=LLM_ID,
        temperature = 0
    )
    test_prompt = "Connection test. Please reply with 'OK'."
    test_llm_connection(llm, test_prompt, TIMEOUT_SEC)

except Exception as e:
    logger.exception(f"Failed to initialize LLM: {e}")
    raise

def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page")
        page_str = f"<page>{page}</page>" if isinstance(page, int) else ""
        formatted.append(
            f"<document><content>{doc.page_content}</content>"
            f"<source>{doc.metadata.get('source', 'unknown')}</source>"
            f"{page_str}</document>"
        )
    return "\n\n".join(formatted)

template = """You are an AI assistant specializing in Question-Answering (QA) tasks within a Retrieval-Augmented Generation (RAG) system. Your mission is to answer the user's QUESTION using only the given CONTEXT (and any included chat history). You must answer in Korean only, concisely, and include important numerical values, technical terms, jargon, and names when they are relevant.

Steps:
1. Read and understand the CONTEXT.
2. Identify information in the CONTEXT that is relevant to the QUESTION.
3. Decide whether the QUESTION can be answered using the CONTEXT.
4. Follow the appropriate output format section below.

Output format when you CAN answer from the CONTEXT:

[Write here the final answer to the QUESTION, concisely in Korean.]

**Source**
- (Write one supporting file name or URL per line.)
- (Add more lines if there are multiple sources.)

Output format when you CANNOT answer from the CONTEXT:

You must output exactly (without quotes and with no additional text):
현재 확인 가능한 자료에 해당 내용이 포함되어 있지 않습니다.

In this case, do not output any **Source** section.

Use only information that appears in the CONTEXT and do not introduce external knowledge or unsupported assumptions.

###
Here is the user's QUESTION:
{question}

Here is the CONTEXT:
{context}

Your final ANSWER to the user's QUESTION:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question", "context"],
)


rag_chain = prompt | llm

def generate(state: State):
    question = state["question"]
    documents = state["documents"]

    if not documents:
        logger.warning("generate() called with no retrieved documents.")
        return {"generation": "현재 확인 가능한 자료에 해당 내용이 포함되어 있지 않습니다.", "messages": [AIMessage("현재 확인 가능한 자료에 해당 내용이 포함되어 있지 않습니다.")]}
    
    logger.info(f"Generating answer for question: '{question}' (docs={len(documents)})")

    try:
        start_time = time.time()
        formatted_context = format_docs(documents)
        response = rag_chain.invoke({"context": formatted_context, "question": question})
        elapsed = time.time() - start_time

        generation = getattr(response, "content", str(response)).strip()
        logger.info(f"Generation complete in {elapsed:.2f}s")

        if not generation:
            return {"generation": "현재 확인 가능한 자료에 해당 내용이 포함되어 있지 않습니다.", "messages": [AIMessage("현재 확인 가능한 자료에 해당 내용이 포함되어 있지 않습니다.")]}

        return {"generation": generation, "messages": [AIMessage(generation)]}

    except Exception as e:
        logger.exception(f"Error during generation for question='{question}': {e}")
        return END