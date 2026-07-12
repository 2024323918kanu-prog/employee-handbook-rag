from sentence_transformers import SentenceTransformer

from app.ingestion.loader import PDFLoader
from app.ingestion.cleaner import TextCleaner
from app.ingestion.chunker import TextChunker
from app.vectordb.database import VectorDatabase


class IndexingPipeline:

    def __init__(self):

        self.loader = PDFLoader()
        self.cleaner = TextCleaner()
        self.chunker = TextChunker()

        print("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.db = VectorDatabase()

    def index_pdf(self, pdf_path):

        print("\n========== INDEXING STARTED ==========\n")

        self.db.reset()

        pages = self.loader.load(pdf_path)

        for page in pages:
            page["text"] = self.cleaner.clean(page["text"])

        chunks = self.chunker.chunk(pages)

        print(f"Chunks Created: {len(chunks)}")

        embeddings = self.embedder.encode(
            [chunk["text"] for chunk in chunks],
            show_progress_bar=True
        )

        embedding_data = []

        for chunk, embedding in zip(chunks, embeddings):
            embedding_data.append(
                {
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "embedding": embedding
                }
            )

        self.db.add_embeddings(embedding_data)

        print(f"Vectors Stored: {self.db.count()}")

        print("\n========== INDEXING COMPLETED ==========\n")