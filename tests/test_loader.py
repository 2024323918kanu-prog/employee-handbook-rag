from app.ingestion.loader import PDFLoader

loader = PDFLoader()

pages = loader.load("data/raw/employee_handbook.pdf")

print("\nExtraction Complete!")
print(f"Pages Extracted: {len(pages)}")

for page in pages:
    print("\n" + "=" * 50)
    print(f"PAGE {page['page']}")
    print("=" * 50)
    print(page["text"])