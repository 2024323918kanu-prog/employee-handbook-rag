import fitz


class PDFLoader:
    """
    Reads a PDF and extracts text page by page.
    """

    def load(self, file_path):
        pages = []

        try:
            document = fitz.open(file_path)

            print("Opened PDF successfully!")
            print(f"Total Pages: {len(document)}")

            for page_number in range(len(document)):
                page = document.load_page(page_number)

                text = page.get_text()

                pages.append({
                    "page": page_number + 1,
                    "text": text.strip()
                })

            document.close()

            return pages

        except Exception as e:
            print("Error:", e)
            return []