class TextChunker:
    """
    Splits cleaned text into overlapping chunks.
    """

    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, pages):

        chunks = []
        seen = set()

        for page in pages:
            text = page["text"]
            start = 0

            while start < len(text):

                end = start + self.chunk_size
                chunk_text = text[start:end].strip()

                # normalize for dedup
                key = f"{page['page']}_{chunk_text[:120].lower()}"

                if key not in seen:
                    chunks.append({
                        "page": page["page"],
                        "text": chunk_text
                    })
                    seen.add(key)

                start += self.chunk_size - self.overlap

        return chunks