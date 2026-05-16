from pydantic import BaseModel, Field
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from dotenv import load_dotenv 

load_dotenv()
embedding_model = get_registry().get("ollama").create(
    name="nomic-embed-text:latest",
    base_url="http://localhost:11434/api/embeddings"  # المسار الافتراضي لـ Ollama على لوكال هوسط
)

EMBEDDING_DIM = 768


class Article(LanceModel):
    doc_id: str
    filepath: str
    filename: str = Field(description="the stem of the file i.e. without the suffix")
    content: str = embedding_model.SourceField()
    embedding: Vector(EMBEDDING_DIM) = embedding_model.VectorField()


class Prompt(BaseModel):
    prompt: str = Field(description="prompt from user, if empty consider prompt as missing")

class RagResponse(BaseModel):
    filename: str = Field(description="filename of the retrieved file without suffix")
    filepath: str = Field(description="absolute path to the retrieved file")
    answer: str = Field(description="answer based on the retrieved file")
