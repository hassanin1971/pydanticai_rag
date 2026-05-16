import lancedb
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from backend.constants import VECTOR_DATABASE_PATH
from backend.data_models import RagResponse

vector_db = lancedb.connect(uri=VECTOR_DATABASE_PATH)

local_model = OllamaModel(
    model_name="llama3.2:latest",
    base_url="http://localhost:11434/v1"  
)

rag_agent = Agent(
    model=local_model,
    retries=3,  
    system_prompt=(
        "You are an expert in rabbit breeds and races, and you know how to distinguish between different rabbits.\n"
        "Always answer based on the retrieved knowledge, but you can mix in your expertise to make the answer more coherent.\n"
        "Don't hallucinate; instead, clearly say you can't answer if the user prompts outside of the retrieved knowledge.\n"
        "Make sure to keep the answer clear and concise, getting to the point directly, max 6 sentences.\n"
        "Also, must explicitly mention which file or filename you used as the source for your answer."
    ),
    output_type=RagResponse,
)


@rag_agent.tool_plain
def retrieve_top_documents(query: str, k: int = 3) -> str:
    """Uses vector search to find the closest k matching documents to the query.

    Args:
        query: The user prompt or question to search for.
        k: Number of documents to retrieve.
    """
    results = vector_db["articles"].search(query=query).limit(k).to_list()
    
    if not results:
        return "No relevant documents found in the knowledge base."

    context_blocks = []
    for idx, doc in enumerate(results):
        block = (
            f"--- Document Source {idx + 1} ---\n"
            f"Filename: {doc.get('filename')}\n"
            f"Filepath: {doc.get('filepath')}\n"
            f"Content: {doc.get('content')}\n"
        )
        context_blocks.append(block)

    return "\n".join(context_blocks)