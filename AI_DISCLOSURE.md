# AI Disclosure & Transparency Report

## 1. Project & Team Information
- **Project Name**: Agentic Code Intelligence
- **Challenge**: Agentic Code Intelligence (CoIR `AppsRetrieval` Text-to-Code Retrieval & Version-Aware Ranking)
- **Repository**: `https://github.com/Adharsh-Arvinth/agentic-code-intelligence.git`

## 2. AI-Assisted Development
- **AI Coding Assistants Used**: Antigravity AI pair programmer (Google Gemini models).
- **Scope of AI Assistance**:
  - Scaffolding modular package structure (`src/preprocessing`, `src/indexing`, `src/retrieval`, `src/reranking`, `src/versioning`, `src/evaluation`).
  - Assisting with regex and Python `ast` structural analyzers (`analyze_code_structure`, `strip_cp_boilerplate`, `_LoopStepGuardTransformer`).
  - Implementing compatibility wrappers for `mteb==2.21.8` (`EncoderProtocol` and `SearchProtocol`) and `PersistentEmbeddingCache`.
  - Writing automated `pytest` unit tests (`44` unit tests across 5 test modules).

## 3. Pretrained Models Used
- **Primary Dense Bi-Encoder (`configs/default.yaml`)**:
  - `BAAI/bge-small-en-v1.5` (33.5M parameters, 384-dimensional embeddings, MIT License). Used with asymmetric query instruction prefixing, Dual-View code indexing (`78%` body + `22%` structural header), and Closed-Form Identity-Anchored Ridge Cross-Modal Alignment (`CrossModalAligner`).
- **Optional / Reference Models Supported in Configuration**:
  - `jinaai/jina-embeddings-v2-base-code` (161M parameters, 768-dimensional embeddings, Apache-2.0 License).
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` (22.7M parameters, optional neural cross-encoder reranker).

## 4. Datasets Used
- **Official Benchmark Dataset**: **CoIR `AppsRetrieval`** (`mteb/AppsRetrieval` / `CoIR-Retrieval/apps` on HuggingFace, dataset revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`).
- **Splits & Partitioning**:
  - **`train` split (`5,000` queries, `8,765` corpus documents)**:
    - `train[0:2000]` (`val_fit`): Used exclusively to fit the closed-form Ridge alignment matrix $W = (Q^T Q + \lambda I)^{-1}(Q^T D + \lambda I)$.
    - `train[2000:2250]` (`val_dev`, 250 held-out validation queries): Used strictly for validation and representation ablation studies (`results/model_comparison.json`).
  - **`test` split (`3,765` queries, `8,765` corpus documents)**:
    - Strictly held out and evaluated via the official `mteb` evaluation pipeline (`results/AppsRetrieval.json`).

## 5. Human Validation & Empirical Verification
- **Empirical Integrity**: All reported metrics in `results/AppsRetrieval.json`, `results/benchmark_summary.json`, `results/model_comparison.json`, and `README.md` are produced by actual local execution on CPU (`mteb==2.21.8`) and verified directly from generated JSON artifacts.
- **Strict Separation of Evaluations**: Official MTEB `AppsRetrieval` `test` split metrics (`NDCG@10 = 0.25591`, `MRR@10 = 0.252327`, `Recall@10 = 0.26746`, `Recall@100 = 0.33732` over `3,765` test queries and `8,765` corpus documents) are documented strictly separately from the 250-query held-out validation ablation study (`6_full_agentic_hybrid_execution_verified`: `NDCG@10 = 0.27920`, `MRR@10 = 0.27099`, `Recall@10 = 0.312`, `Recall@100 = 0.392`).
- **Automated Verification**: Validated via `44/44` passing `pytest` unit tests covering evaluation metrics, preprocessing, hybrid fusion, version-aware indexing/retrieval (`v1.0` / `v2.0` registration, metadata persistence, version filtering, and deduplication), and MTEB protocol compliance.
