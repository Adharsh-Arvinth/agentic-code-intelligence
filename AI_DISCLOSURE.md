# AI Disclosure & Transparency Report

## 1. AI Coding Assistance
- **AI Coding Assistants**: Antigravity AI pair programmer / Google Gemini coding models.
- **Scope of AI Assistance**: Scaffold generation, test case implementation, multi-stage pipeline implementation, and bug fixes for HuggingFace/Transformers library compatibility.

## 2. External Pretrained Models Used
- **Dense Embedding Model**: `BAAI/bge-small-en-v1.5` (33.5M parameters, 512 context length, 384 embedding dimensions).
- **Optional High-Accuracy / Reference Model**: `jinaai/jina-embeddings-v2-base-code` (161M parameters, 8192 context length).
- **Cross-Encoder Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22.7M parameters).

## 3. Datasets Used
- **Dataset**: **CoIR AppsRetrieval** (Text-to-Code retrieval benchmark derived from competitive programming problem descriptions and Python solutions).
- **Source**: Loaded automatically via HuggingFace `datasets` library (`mteb/AppsRetrieval` / `CoIR-Retrieval/apps`).

## 4. Human Validation & Automated Testing
- **Validation**: Every component was empirically verified through automated pytest test suites (42 passing unit test cases covering metrics, preprocessing, retrieval, indexing, and versioning).
- **Empirical Measurement**: Benchmark scores (NDCG@10, MRR@10, Recall@10, MAP@10) were generated through real inference runs using MTEB and recorded directly from system execution logs.
