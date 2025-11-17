from typing import Annotated
from langgraph.graph import MessagesState

class State(MessagesState):
    question: Annotated[str, "User question"]
    generation: Annotated[str, "LLM generated answer"]
    documents: Annotated[list[str], "List of documents"]