from app.llm.ollama_client import OllamaClient

llm = OllamaClient()

answer = llm.generate(
    "Explain machine learning in one sentence."
)

print(answer)