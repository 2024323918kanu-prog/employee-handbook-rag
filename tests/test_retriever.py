from app.retrieval.retriever import Retriever

retriever = Retriever()

results = retriever.search(
    "What is the dress code?"
)

for chunk in results:

    print("=" * 60)

    print("PAGE:", chunk["page"])

    print()

    print(chunk["text"][:300])