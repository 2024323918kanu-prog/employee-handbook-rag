# 📘 Employee Handbook RAG Assistant

An AI-powered Employee Handbook Assistant built using Retrieval-Augmented Generation (RAG). The application answers employee questions by retrieving relevant information from an employee handbook using Hybrid Retrieval (Semantic Search + BM25) and CrossEncoder reranking before generating responses with Llama 3.2 running locally through Ollama.

## Preview

### Home

![Home](assets/home.png)

### Chat

![Chat](assets/chat.png)

## Features

- Hybrid Retrieval (Semantic Search + BM25)
- CrossEncoder Reranking
- ChromaDB Vector Database
- SentenceTransformer Embeddings
- Local LLM using Ollama (Llama 3.2 3B)
- Source Attribution
- Performance Metrics
- Automated Retrieval Evaluation Framework (Precision@K, Recall@K, MRR)
- Streamlit Web Interface

## Tech Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Llama 3.2 3B (Ollama) |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Reranker | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| PDF Processing | PyMuPDF |
| ML Framework | Sentence Transformers |

## Architecture

```text
                  Employee Handbook (PDF)
                            │
                            ▼
                     PDF Loader (PyMuPDF)
                            │
                            ▼
                     Text Cleaning
                            │
                            ▼
                     Text Chunking
                            │
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
SentenceTransformer                   BM25 Index
          ▼                                   ▼
      ChromaDB                     Keyword Search
          └──────────────┬────────────────────┘
                         ▼
                 Hybrid Retrieval
                         ▼
              CrossEncoder Reranker
                         ▼
                  Prompt Builder
                         ▼
              Llama 3.2 (Ollama)
                         ▼
               Streamlit Web App
```

## Performance & Evaluation

Retrieval quality was measured using a hand-labeled evaluation set of **35 queries** spanning 8 handbook categories (leave policy, conduct, compensation, IT/security, benefits, employment basics, termination, and workplace safety), each mapped to ground-truth chunk IDs in the indexed corpus (145 chunks, 34 pages).

Three retrieval configurations were benchmarked head-to-head at K=5:

| Configuration | Retrieval Accuracy | Precision@5 | Recall@5 | MRR | Avg. Retrieval Latency |
|---|---|---|---|---|---|
| Semantic Search Only | **94.3%** | 0.240 | **90.0%** | **0.815** | 0.06s |
| Hybrid (Semantic + BM25) | 88.6% | 0.223 | 82.9% | 0.793 | 0.75s |
| Hybrid + CrossEncoder Rerank | 88.6% | 0.229 | 84.3% | 0.783 | 0.68s |

**Key finding:** on this corpus — a single, narrow-domain, 145-chunk document — semantic search alone outperforms hybrid retrieval. During evaluation, a bug was identified in the fusion logic: BM25 scores were normalized against each query's own top result rather than a fixed reference point, which let low-confidence keyword matches receive artificially high fusion weight. Fixing this (using a fixed, seeded, corpus-wide reference score) improved hybrid Recall@5 from 78.6% to 82.9% and MRR from 0.731 to 0.793. Even after the fix, however, semantic search remained stronger overall — consistent with hybrid retrieval's known strength being most impactful on large, lexically diverse corpora (e.g. multi-document knowledge bases, technical/legal text with rare terms) rather than a single handbook with a narrow, consistent vocabulary. CrossEncoder reranking still provides a measurable lift within hybrid retrieval (Recall@5: 82.9% → 84.3%), confirming its value as a refinement step even where fusion itself underperforms.

**Precision@5 note:** precision appears low (~0.22–0.24) across all configurations by design, not due to poor retrieval — most queries have only 1–2 genuinely relevant chunks out of the 5 retrieved, which mathematically caps precision regardless of ranking quality. Recall@5 and MRR are the more informative metrics for this evaluation.

### Reproducing these results

```bash
python -m app.evaluation.run_evaluation
```

This runs all 35 evaluation queries through each retrieval configuration and prints a comparison table. Ground truth lives in `app/evaluation/eval_dataset.json`; per-query regression analysis is available via:

```bash
python -m app.evaluation.diagnose_fusion
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/employee-handbook-rag.git
cd employee-handbook-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## Project Structure

```text
employee-handbook-rag/
│
├── app/
│   ├── ingestion/
│   ├── llm/
│   ├── retrieval/
│   ├── services/
│   ├── evaluation/
│   └── vectordb/
│
├── data/
│   └── raw/
│
├── tests/
│
├── streamlit_app.py
├──  index_handbook.py
├── requirements.txt
└── README.md
```

## Sample Questions

- What are the working hours?
- Can I work from home?
- What is the dress code?
- How many annual leave days do I get?
- Is smoking allowed?
- What benefits do full-time employees receive?
- What is the overtime policy?

## Future Improvements

- Conversation memory
- Multi-document support
- PDF upload through the UI
- Response streaming
- Citation highlighting
- Docker deployment
- Authentication
- Average Generation Latency benchmarking (LLM response time) alongside retrieval metrics
- Reciprocal Rank Fusion (RRF) as an alternative to weighted-score fusion

## License

This project is developed for learning and portfolio purposes.

## How it Works

1. The employee handbook PDF is loaded and cleaned.
2. The document is split into smaller chunks.
3. Chunks are converted into embeddings using Sentence Transformers.
4. Embeddings are stored in ChromaDB.
5. User questions are searched using Hybrid Retrieval:
   - Semantic Search
   - BM25 Keyword Search
6. Retrieved chunks are reranked using a CrossEncoder.
7. The top-ranked context is passed to Llama 3.2 through Ollama.
8. The assistant generates an answer along with the relevant source pages.
9. Retrieval quality is continuously validated against a 35-query labeled evaluation set (see Performance & Evaluation above).
