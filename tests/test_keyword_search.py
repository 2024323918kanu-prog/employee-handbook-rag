from app.ingestion.loader import PDFLoader
from app.ingestion.cleaner import TextCleaner
from app.ingestion.chunker import TextChunker
from app.retrieval.keyword_search import KeywordSearch

# Load PDF
loader = PDFLoader()
pages = loader.load("data/raw/employee_handbook.pdf")

# Clean text
cleaner = TextCleaner()

cleaned_pages = []

for page in pages:
    cleaned_pages.append({
        "page": page["page"],
        "text": cleaner.clean(page["text"])
    })

# Chunk text
chunker = TextChunker()

chunks = chunker.chunk(cleaned_pages)

# Extract only the text for BM25
documents = chunks

# Build BM25 index
keyword_search = KeywordSearch()
keyword_search.build_index(documents)

# Ask a question
query = input("Enter your question: ")

results = keyword_search.search(query)

print("\nTop Results:\n")

for i, result in enumerate(results, start=1):
    print("=" * 60)
    print(f"Result {i}")
    print(f"Page: {result['page']}")
    print()
    print(result["text"][:400])
    print()