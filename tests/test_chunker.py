from app.ingestion.loader import PDFLoader
from app.ingestion.cleaner import TextCleaner
from app.ingestion.chunker import TextChunker

loader = PDFLoader()
cleaner = TextCleaner()
chunker = TextChunker()

pages = loader.load("data/raw/employee_handbook.pdf")

for page in pages:
    page["text"] = cleaner.clean(page["text"])

chunks = chunker.chunk(pages)

print(f"\nTotal Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print("\n" + "=" * 60)
    print(f"CHUNK {i}")
    print("=" * 60)
    print(chunk["text"])