from app.ingestion.loader import PDFLoader
from app.ingestion.cleaner import TextCleaner

loader = PDFLoader()
cleaner = TextCleaner()

pages = loader.load("data/raw/employee_handbook.pdf")

for page in pages:
    print("=" * 60)
    print("BEFORE CLEANING")
    print("=" * 60)
    print(page["text"])

    cleaned = cleaner.clean(page["text"])

    print("\n")
    print("=" * 60)
    print("AFTER CLEANING")
    print("=" * 60)
    print(cleaned)