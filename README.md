# Agentic Code Intelligence

A production-quality, high-performance **Code Retrieval System** built for the official **CoIR AppsRetrieval** benchmark and evaluated using the **MTEB** framework.

Given a natural-language query describing programmatic intent or algorithmic requirements, the system retrieves and ranks Python code solutions from a large corpus of competitive programming solutions.

---

## Architecture

```
Natural Language Query
        │
        ▼
┌─────────────────────────┐
│  Query Preprocessing    │  Whitespace normalization, identifier extraction
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Stage 1: Semantic      │  Dense vector retrieval (FAISS IndexFlatIP)
│  Retrieval              │  Model: BAAI/bge-small-en-v1.5
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Stage 2: Lexical       │  BM25 retrieval with code-aware tokenization
│  Retrieval              │  (snake_case & camelCase splitting)
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Stage 3: Hybrid Fusion │  Reciprocal Rank Fusion (RRF)
│  (Top Candidate Set)    │  (Combines Dense + Lexical rankings)
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Stage 4: Cross-Encoder │  Cross-Encoder Reranking
│  Reranking              │  Model: cross-encoder/ms-marco-MiniLM-L-6-v2
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Final Ranked Results   │  Ranked code snippets, relevance scores, metadata
└─────────────────────────┘
```

---

## Features

- **Semantic Vector Search**: High-dimensional dense code retrieval powered by FAISS vector indexing.
- **Lexical BM25 Search**: Code-aware tokenization splitting camelCase and snake_case identifiers.
- **Reciprocal Rank Fusion (RRF)**: Combines dense semantic rankings and sparse BM25 rankings without score normalization issues.
- **Cross-Encoder Reranking**: Multi-stage precision refinement using MS-MARCO cross-encoders.
- **Version-Aware Retrieval**: Tag, isolate, search, and manage code snippet indexes by version tags.
- **CPU & GPU Support**: Runs efficiently on CPU without CUDA hardware requirements.
- **Official MTEB Integration**: Evaluated via the official MTEB benchmark framework.

---

## Models

| Component | Default Model | Parameters | Context | Description |
|-----------|---------------|------------|---------|-------------|
| **Default Embedding** | `BAAI/bge-small-en-v1.5` | 33.5M | 512 | Fast, lightweight dense retrieval model |
| **High-Capacity Option** | `jinaai/jina-embeddings-v2-base-code` | 161M | 8192 | Long-context code-specialized embedding model |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22.7M | 512 | Second-stage cross-encoder ranker |

---

## Installation

```bash
git clone https://github.com/Adharsh-Arvinth/agentic-code-intelligence.git
cd agentic-code-intelligence
py -m pip install -r requirements.txt
```

### System Requirements
- **Python**: 3.10+
- **RAM**: 4GB minimum (8GB recommended)
- **GPU**: Optional (CPU execution is fully supported)

---

## Dataset

The system uses the **CoIR AppsRetrieval** dataset loaded directly from HuggingFace (`mteb/AppsRetrieval` / `CoIR-Retrieval/apps`):
- **Task**: Text-to-Code retrieval (Problem statement to Python code solution)
- **Queries**: 3,765 test problem descriptions
- **Corpus**: 8,765 Python code solutions
- **Relevance**: Binary relevance labels

---

## Usage

### 1. Build Index
Pre-build and cache FAISS and BM25 indexes:
```bash
py -m src index
```

### 2. Natural Language Query Retrieval
Run queries directly from the CLI:
```bash
py -m src retrieve --query "Given an integer array nums, return true if any value appears at least twice"
```

Options:
- `--top-k 10`: Number of results to return (default: 10)
- `--method hybrid`: Retrieval method (`hybrid`, `semantic`, or `lexical`)
- `--no-rerank`: Disable cross-encoder reranking
- `--sample N`: Fast execution on a subset of N corpus documents

### 3. Version-Aware Retrieval
Index and retrieve from specific corpus version tags:
```bash
py -m src retrieve --query "binary search array" --version v1.0
```

---

## Actual Benchmark Results

All metrics below are measured from empirical execution runs on the **CoIR AppsRetrieval** test set.

### Official MTEB AppsRetrieval Benchmark Results
*(Evaluated via `py -m src mteb_eval` on full test split)*

| Metric | Score | JSON Output Location |
|--------|-------|----------------------|
| **NDCG@10** | `0.0514` | `results/BAAI__bge-small-en-v1.5/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/AppsRetrieval.json` |
| **MRR@10** | `0.0437` | `results/BAAI__bge-small-en-v1.5/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/AppsRetrieval.json` |
| **Recall@10** | `0.0765` | `results/BAAI__bge-small-en-v1.5/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/AppsRetrieval.json` |
| **MAP@10** | `0.0437` | `results/BAAI__bge-small-en-v1.5/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/AppsRetrieval.json` |

### Method Comparison (Empirical Experiment Runs)
*(Evaluated via `py -m src evaluate` on test queries)*

| Method | NDCG@10 | MRR@10 | Recall@10 |
|--------|---------|--------|-----------|
| **Lexical Only (BM25)** | 0.0163 | 0.0192 | 0.0200 |
| **Semantic Only (Dense)** | 0.0632 | 0.0618 | 0.0800 |
| **Hybrid (RRF Dense + BM25)** | **0.0718** | **0.0619** | **0.1100** |

*Hybrid Reciprocal Rank Fusion (RRF) delivers the highest NDCG@10 (0.0718) and Recall@10 (0.1100), outperforming single-retriever baselines.*

---

## Project Structure

```
agentic-code-intelligence/
├── configs/
│   └── default.yaml             # System configuration parameters
├── src/
│   ├── __init__.py               # Transformers 5.x compatibility patches
│   ├── __main__.py              # CLI entry point dispatcher
│   ├── evaluate.py              # Custom experiment benchmark script
│   ├── index.py                 # Corpus indexing script
│   ├── mteb_eval_cli.py         # Official MTEB evaluation runner
│   ├── retrieve.py              # CLI retrieval runner
│   ├── evaluation/
│   │   ├── experiment_runner.py # Automated experiment runner
│   │   ├── metrics.py           # Ranking metrics (NDCG, MRR, MAP, Recall)
│   │   └── mteb_wrapper.py      # MTEB-compatible model wrapper
│   ├── indexing/
│   │   ├── embedding_index.py   # FAISS dense vector indexer
│   │   ├── index_manager.py     # Unified index manager
│   │   └── lexical_index.py     # Code-aware BM25 indexer
│   ├── pipeline/
│   │   └── pipeline.py          # End-to-end Retrieval Pipeline
│   ├── preprocessing/
│   │   ├── code_preprocessor.py # Code tokenization & structural extraction
│   │   └── query_preprocessor.py# Query normalization & keyword extraction
│   ├── reranking/
│   │   └── cross_encoder.py     # Cross-encoder reranker
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # RRF hybrid retriever
│   │   ├── lexical_retriever.py # BM25 retriever
│   │   └── semantic_retriever.py# Dense FAISS retriever
│   ├── utils/
│   │   ├── config.py            # Config loader
│   │   ├── data_loader.py       # Dataset loader
│   │   └── logging_utils.py     # Formatter and performance logging
│   └── versioning/
│       ├── dataset_versioner.py # Dataset version manager
│       └── index_versioner.py   # Index version manager
├── tests/                       # Complete pytest unit test suite (42 tests)
├── AI_DISCLOSURE.md             # AI transparency disclosure
├── README.md                    # System documentation
└── requirements.txt             # Python dependencies
```

---

## Automated Test Suite

Run the full unit test suite:
```bash
py -m pytest tests/ -v
```

Output:
```text
============================== 42 passed in 24.64s ==============================
```

---

## Hackathon Submission Checklist

- [x] **Source Code**: Production-ready Python codebase (`src/`)
- [x] **README**: Tested reproduction steps and real benchmark results
- [x] **AI Disclosure**: Included in `AI_DISCLOSURE.md`
- [x] **Test Suite**: 42 passing unit test cases in `tests/`
- [x] **MTEB Benchmark Output**: Generated and saved in `results/`
- [x] **SDK / API**: Modular Python package (`src.pipeline.RetrievalPipeline`)
- [ ] **Presentation**: (To be created manually)
- [ ] **Video**: (To be created manually)
- [x] **APK**: Not Applicable (Backend/CLI Python code retrieval engine)

---

## License

MIT License.
