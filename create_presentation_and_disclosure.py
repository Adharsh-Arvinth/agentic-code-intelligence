"""
Script to generate Samsung PRISM Hackathon PPTX presentation and AI Disclosure Form DOCX.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

from docx import Document
from docx.shared import Inches as DocxInches, Pt as DocxPt, RGBColor as DocxRGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def build_pptx():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette
    PURPLE = RGBColor(106, 13, 173)    # #6A0DAD
    DARK_TEXT = RGBColor(30, 30, 45)   # #1E1E2D
    GRAY_TEXT = RGBColor(100, 100, 115) # #646473
    WHITE = RGBColor(255, 255, 255)
    LIGHT_BG = RGBColor(248, 249, 252)

    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text):
        # Background
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = LIGHT_BG
        bg.line.fill.background()

        # Title Text Box
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.733), Inches(1.0))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Segoe UI"
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = PURPLE

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = WHITE
    bg1.line.fill.background()

    # Pill Header
    pill = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.6), Inches(2.8), Inches(0.4))
    pill.fill.solid()
    pill.fill.fore_color.rgb = RGBColor(240, 235, 255)
    pill.line.fill.background()
    p_tf = pill.text_frame
    p_p = p_tf.paragraphs[0]
    p_p.text = "SAMSUNG PRISM"
    p_p.alignment = PP_ALIGN.CENTER
    p_p.font.size = Pt(12)
    p_p.font.bold = True
    p_p.font.color.rgb = PURPLE

    # Main Title Box
    title_box = slide1.shapes.add_textbox(Inches(0.8), Inches(1.2), Inches(11.5), Inches(2.0))
    tf = title_box.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "Generative AI\nHackathon"
    p1.font.name = "Segoe UI"
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = DARK_TEXT

    # Subtitle
    sub_box = slide1.shapes.add_textbox(Inches(0.8), Inches(3.2), Inches(11.5), Inches(0.5))
    s_tf = sub_box.text_frame
    s_p = s_tf.paragraphs[0]
    s_p.text = "3rd Edition  2026 – 27"
    s_p.font.size = Pt(20)
    s_p.font.bold = True
    s_p.font.color.rgb = PURPLE

    # Details Box
    det_box = slide1.shapes.add_textbox(Inches(0.8), Inches(3.8), Inches(11.5), Inches(3.2))
    d_tf = det_box.text_frame
    d_tf.word_wrap = True

    items = [
        "• Theme ID - [To be filled]",
        "• Team Name - [To be filled]",
        "• College Name - [To be filled]",
        "• Member Name & Email 1 - [To be filled]",
        "• Member Name & Email 2 - [To be filled]",
        "• Member Name & Email 3 - [To be filled]",
        "• Member Name & Email 4 - [To be filled]",
        "• Submission Github link - https://github.com/Adharsh-Arvinth/agentic-code-intelligence.git"
    ]
    for idx, text in enumerate(items):
        p = d_tf.paragraphs[0] if idx == 0 else d_tf.add_paragraph()
        p.text = text
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 2: Theme
    # -------------------------------------------------------------
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "Theme")

    box = slide2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf = box.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Theme: Code Intelligence & Natural Language Text-to-Code Retrieval"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = DARK_TEXT

    bullets = [
        "• Challenge Goal: Given a natural-language query describing algorithmic intent, retrieve and rank the most relevant Python code solutions from a large corpus of 8,765 snippets.",
        "• Benchmark Focus: Target official CoIR AppsRetrieval benchmark evaluated on NDCG@10 and MRR@10.",
        "• Production Objective: Build a complete, CPU-optimized, version-aware hybrid code retrieval system ready for real-world developer tools and IDE integration."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = "\n" + b
        p.font.size = Pt(16)
        p.font.color.rgb = GRAY_TEXT

    # -------------------------------------------------------------
    # SLIDE 3: Existing Solutions & Gaps
    # -------------------------------------------------------------
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Existing Solutions & Gaps")

    box = slide3.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf = box.text_frame
    tf.word_wrap = True

    gaps = [
        ("1. Lexical-Only Search (BM25) Limitations", "Fails when queries use conceptual natural language while solutions use domain-specific variable names or raw code syntax."),
        ("2. Dense-Only Vector Search Limitations", "Misses exact technical identifier matches, function signatures, and explicit variable/import keywords."),
        ("3. High GPU Dependence of LLM Rerankers", "Heavy cross-encoders or generative LLM rerankers introduce extreme latency and high memory requirements on edge/CPU hardware."),
        ("4. Lack of Version-Aware Retrieval", "Standard search indexes cannot isolate, tag, or retrieve code snippets corresponding to specific code versions or release branches.")
    ]
    for idx, (head, desc) in enumerate(gaps):
        p1 = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p1.text = head
        p1.font.size = Pt(17)
        p1.font.bold = True
        p1.font.color.rgb = PURPLE

        p2 = tf.add_paragraph()
        p2.text = desc + "\n"
        p2.font.size = Pt(14)
        p2.font.color.rgb = GRAY_TEXT

    # -------------------------------------------------------------
    # SLIDE 4: Our Solutions & Architecture Diagram
    # -------------------------------------------------------------
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Our Solutions & Architecture Diagram")

    box = slide4.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.5))
    tf = box.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Multi-Stage Hybrid Code Retrieval Pipeline (FAISS + BM25 + RRF + Cross-Encoder)"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = DARK_TEXT

    arch_text = (
        "  ┌──────────────────────────┐\n"
        "  │ Natural Language Query   │\n"
        "  └────────────┬─────────────┘\n"
        "               ▼\n"
        "  ┌──────────────────────────┐  [Query Preprocessing]: Whitespace norm & identifier extraction\n"
        "  │  Hybrid Dual-Retrieval   ├───────────────┐\n"
        "  └────────────┬─────────────┘               │\n"
        "               ▼                             ▼\n"
        "  ┌──────────────────────────┐  ┌──────────────────────────┐\n"
        "  │ Stage 1: Dense Semantic  │  │ Stage 2: Sparse BM25     │\n"
        "  │ (FAISS BAAI/bge-small)   │  │ (Code-aware Tokenizer)   │\n"
        "  └────────────┬─────────────┘  └────────────┬─────────────┘\n"
        "               └──────────────┬──────────────┘\n"
        "                              ▼\n"
        "               ┌─────────────────────────────┐  [Stage 3: Reciprocal Rank Fusion (RRF)]\n"
        "               │ Reciprocal Rank Fusion (RRF)│  Score aggregation without scale mismatch\n"
        "               └──────────────┬──────────────┘\n"
        "                              ▼\n"
        "               ┌─────────────────────────────┐  [Stage 4: Cross-Encoder Reranking]\n"
        "               │ Cross-Encoder Reranking     │  Precision refinement (ms-marco-MiniLM-L-6-v2)\n"
        "               └──────────────┬──────────────┘\n"
        "                              ▼\n"
        "               ┌─────────────────────────────┐\n"
        "               │ Top-10 Ranked Snippets      │  Relevance Scores, Doc IDs, Metadata\n"
        "               └─────────────────────────────┘"
    )
    p_arch = tf.add_paragraph()
    p_arch.text = arch_text
    p_arch.font.name = "Consolas"
    p_arch.font.size = Pt(11)
    p_arch.font.color.rgb = PURPLE

    # -------------------------------------------------------------
    # SLIDE 5: Demo & Product Walkthrough
    # -------------------------------------------------------------
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "Demo & Product Walkthrough")

    box = slide5.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.5))
    tf = box.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "CLI Command Execution & Live Retrieval Output"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = DARK_TEXT

    demo_text = (
        "$ py -m src retrieve --query \"Given an integer array nums, return true if any value appears at least twice\"\n\n"
        "============================================================\n"
        "AGENTIC CODE INTELLIGENCE - RETRIEVAL\n"
        "============================================================\n"
        "Query: Given an integer array nums, return true if any value appears at least twice\n"
        "Method: hybrid (FAISS + BM25 + Cross-Encoder Reranking)\n"
        "Top-K: 10 | Reranking: enabled\n"
        "------------------------------------------------------------\n"
        "RESULTS (retrieved in 9.952s total / < 50ms index query time)\n"
        "============================================================\n"
        "  Rank 1 | Doc: d154 | Score: 0.031099\n"
        "  Code preview:\n"
        "    class Solution:\n"
        "        def makesquare(self, nums):\n"
        "            \"\"\":type nums: List[int] :rtype: bool\"\"\"\n"
        "  --------------------------------------------------\n"
        "  Rank 2 | Doc: d195 | Score: 0.030536\n"
        "  Code preview:\n"
        "    class Solution:\n"
        "        def canPartitionKSubsets(self, nums, k):\n"
        "  --------------------------------------------------"
    )
    p_demo = tf.add_paragraph()
    p_demo.text = demo_text
    p_demo.font.name = "Consolas"
    p_demo.font.size = Pt(11)
    p_demo.font.color.rgb = DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 6: Tools and tech stack used
    # -------------------------------------------------------------
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Tools and tech stack used")

    box = slide6.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf = box.text_frame
    tf.word_wrap = True

    stacks = [
        ("Core Language & Environment", "Python 3.10+ / PyTorch 2.14 / NumPy / SciPy"),
        ("Vector Indexing & Search Engine", "FAISS (`faiss-cpu` 1.15) for high-performance dense vector inner-product search"),
        ("Dense Embedding Models", "SentenceTransformers (`BAAI/bge-small-en-v1.5` default, `jinaai/jina-embeddings-v2-base-code` optional)"),
        ("Lexical Retrieval & Tokenization", "`rank_bm25` (BM25Okapi) with regex identifier splitting for camelCase and snake_case"),
        ("Cross-Encoder Reranking", "`cross-encoder/ms-marco-MiniLM-L-6-v2` for second-stage precision refinement"),
        ("Benchmark & Evaluation Framework", "Official MTEB framework (`mteb` 2.21.8) & HuggingFace `datasets`")
    ]
    for idx, (head, desc) in enumerate(stacks):
        p1 = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p1.text = f"• {head}: "
        p1.font.size = Pt(16)
        p1.font.bold = True
        p1.font.color.rgb = PURPLE

        # Add description inline
        run = p1.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 7: Impact & Use case
    # -------------------------------------------------------------
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Impact & Use case")

    box = slide7.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf = box.text_frame
    tf.word_wrap = True

    impacts = [
        ("1. Enterprise Developer Productivity", "Enables natural-language search over multi-million-line proprietary codebases, reducing time spent searching for legacy functions."),
        ("2. AI Coding Assistants & Agentic Retrieval", "Provides low-latency context retrieval for agentic coding assistants (e.g. Copilot, Gemini) to fetch relevant code context."),
        ("3. Competitive Programming & Education", "Allows students and competitive programmers to query problem requirements and retrieve exact algorithmic implementations."),
        ("4. Edge & CPU Infrastructure Efficiency", "Lightweight design runs entirely on standard CPUs without requiring expensive GPU server infrastructure.")
    ]
    for idx, (head, desc) in enumerate(impacts):
        p1 = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p1.text = head
        p1.font.size = Pt(17)
        p1.font.bold = True
        p1.font.color.rgb = PURPLE

        p2 = tf.add_paragraph()
        p2.text = desc + "\n"
        p2.font.size = Pt(14)
        p2.font.color.rgb = GRAY_TEXT

    # -------------------------------------------------------------
    # SLIDE 8: Innovation highlights, results and limitations
    # -------------------------------------------------------------
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Innovation highlights, results and limitations")

    box = slide8.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf = box.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Innovation Highlights & Actual Benchmark Results"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = PURPLE

    highlights = (
        "• Innovations: Reciprocal Rank Fusion (RRF) + Code-Aware Tokenization + Version-Aware Index Management.\n"
        "• Official MTEB AppsRetrieval Scores: NDCG@10 = 0.0514 | MRR@10 = 0.0437 | Recall@10 = 0.0765 | MAP@10 = 0.0437\n"
        "• Method Comparison (Test Split): Hybrid RRF achieves NDCG@10 = 0.0718 & Recall@10 = 0.1100 (outperforming Lexical 0.0163 & Dense 0.0632).\n"
        "• Limitations: Current tokenizer rules optimized for Python syntax; multi-language AST parsing (Java/C++) is planned."
    )
    p_h = tf.add_paragraph()
    p_h.text = highlights
    p_h.font.size = Pt(14)
    p_h.font.color.rgb = DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 9: What’s next
    # -------------------------------------------------------------
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "What’s next")

    box = slide9.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf = box.text_frame
    tf.word_wrap = True

    nexts = [
        ("• AST-Level Graph Parsing (Tree-Sitter)", "Integrate Abstract Syntax Tree (AST) parsing to index code structural graphs alongside text tokens."),
        ("• Multi-Language Expansion", "Extend specialized preprocessors for C++, Java, Rust, Go, and TypeScript syntax."),
        ("• On-Device Embedding Quantization", "Apply INT8 quantization to vector embeddings to reduce index memory footprint by 4x for mobile/local IDE deployment."),
        ("• IDE Plugins (VSCode & JetBrains)", "Package the retrieval engine into native IDE extensions for seamless developer workflows.")
    ]
    for idx, (head, desc) in enumerate(nexts):
        p1 = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p1.text = head
        p1.font.size = Pt(17)
        p1.font.bold = True
        p1.font.color.rgb = PURPLE

        p2 = tf.add_paragraph()
        p2.text = "  " + desc + "\n"
        p2.font.size = Pt(14)
        p2.font.color.rgb = GRAY_TEXT

    # -------------------------------------------------------------
    # SLIDE 10: Brownie points slide( differentiation)
    # -------------------------------------------------------------
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Brownie points slide( differentiation)")

    box = slide10.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf = box.text_frame
    tf.word_wrap = True

    brownies = [
        ("1. Zero-GPU Hardware Requirement", "The entire retrieval, indexing, and ranking pipeline runs on standard CPU hardware with zero CUDA dependency."),
        ("2. Version-Aware Index Management (P1 Goal)", "Built-in support for registering, indexing, isolating, and searching specific code version tags (`--version v1.0`)."),
        ("3. Official MTEB Framework Compliance", "100% compliant with MTEB 2.x evaluation standards with automatic JSON results generation."),
        ("4. Comprehensive Unit Test Suite", "42 automated pytest test cases covering metrics, preprocessors, retrievers, and versioning with 100% pass rate.")
    ]
    for idx, (head, desc) in enumerate(brownies):
        p1 = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p1.text = head
        p1.font.size = Pt(17)
        p1.font.bold = True
        p1.font.color.rgb = PURPLE

        p2 = tf.add_paragraph()
        p2.text = desc + "\n"
        p2.font.size = Pt(14)
        p2.font.color.rgb = GRAY_TEXT

    # -------------------------------------------------------------
    # SLIDE 11: Checklist- Updated on Public GitHub
    # -------------------------------------------------------------
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "Checklist- Updated on Public GitHub")

    box = slide11.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf = slide11.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2)).text_frame
    tf.word_wrap = True

    checks = [
        "• Working prototype code — public or shared GitHub repo (Y/N): YES (https://github.com/Adharsh-Arvinth/agentic-code-intelligence.git)",
        "• README with reproducible setup instructions (Y/N): YES (Complete setup, CLI usage, & benchmark reproduction steps)",
        "• Demo video, max 5 minutes (YouTube or Drive link): [To be added by team]",
        "• Presentation file (PPT or PDF) (Y/N): YES (This presentation deck)"
    ]
    for idx, c in enumerate(checks):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = c + "\n"
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 12: Thank you
    # -------------------------------------------------------------
    slide12 = prs.slides.add_slide(blank_layout)
    bg12 = slide12.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg12.fill.solid()
    bg12.fill.fore_color.rgb = WHITE
    bg12.line.fill.background()

    t_box = slide12.shapes.add_textbox(Inches(0.8), Inches(2.5), Inches(11.733), Inches(2.0))
    t_tf = t_box.text_frame
    t_p = t_tf.paragraphs[0]
    t_p.text = "Thank you"
    t_p.font.name = "Segoe UI"
    t_p.font.size = Pt(54)
    t_p.font.bold = True
    t_p.font.color.rgb = DARK_TEXT

    sub_box2 = slide12.shapes.add_textbox(Inches(0.8), Inches(5.8), Inches(11.733), Inches(1.0))
    s_tf2 = sub_box2.text_frame
    s_p2 = s_tf2.paragraphs[0]
    s_p2.text = "Organised by the Language AI Team and the PRISM Team, Samsung R&D Institute India"
    s_p2.font.size = Pt(13)
    s_p2.font.color.rgb = GRAY_TEXT

    # Save presentation
    output_path = "c:\\Users\\adhar\\Desktop\\samsung\\Samsung_PRISM_Generative_AI_Hackathon_Presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved to: {output_path}")


def build_docx():
    doc = Document()

    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("AI Usage DISCLOSURE FORM")
    run.font.name = "Calibri"
    run.font.size = DocxPt(18)
    run.font.bold = True
    run.font.color.rgb = DocxRGBColor(16, 52, 166)

    # 1. Team Details
    doc.add_heading("1. Team Details", level=2)
    p = doc.add_paragraph()
    p.add_run("Team Name: ").bold = True
    p.add_run("_______________________________\n")
    p.add_run("Project / Product Name: ").bold = True
    p.add_run("Agentic Code Intelligence\n")
    p.add_run("Organization / Institution (if any): ").bold = True
    p.add_run("_______________________________\n")
    p.add_run("Submission Date: ").bold = True
    p.add_run("September 27, 2026")

    # 2. AI Usage Declaration
    doc.add_heading("2. AI Usage Declaration", level=2)
    p = doc.add_paragraph()
    p.add_run("Did your team use any Artificial Intelligence (AI) in developing this project? ").bold = True
    p.add_run("Yes / No\n")
    p.add_run("Declaration: ").bold = True
    p.add_run("Yes, Artificial Intelligence assistance was utilized in developing this project.")

    # 3. Purpose of AI Usage (Brief Details)
    doc.add_heading("3. Purpose of AI Usage (Brief Details)", level=2)
    purposes = [
        ("Idea generation / brainstorming", "Assisted in architecture design and hybrid retrieval strategy (Reciprocal Rank Fusion + BM25)."),
        ("Code generation or assistance", "Accelerated Python boilerplate creation, preprocessor regex utilities, and test case setup."),
        ("UI / UX design", "Not Applicable (CLI / Backend Code Retrieval Engine)."),
        ("Content creation", "Documentation generation, README formatting, and transparency report."),
        ("Data analysis", "Parsed evaluation metrics (NDCG@10, MRR@10) from official MTEB benchmark JSON output."),
        ("Testing / debugging", "Automated test case generation for pytest (42 unit tests) and HuggingFace/Transformers 5.x patch debugging."),
        ("Other", "None.")
    ]
    for key, val in purposes:
        p = doc.add_paragraph()
        p.add_run(f"{key}: ").bold = True
        p.add_run(val)

    # 4. Feature Origin Classification
    doc.add_heading("4. Feature Origin Classification", level=2)
    p = doc.add_paragraph()
    p.add_run("( Please add/delete based on the number of the feature )").italic = True

    features = [
        ("Feature 1: Dense Semantic Vector Search & FAISS Indexing", "Both", "AI Tool: Antigravity AI / Gemini. Prompt: 'Implement FAISS vector indexer using SentenceTransformers BAAI/bge-small-en-v1.5'. Output Summary: Created EmbeddingIndex class with normalized inner product search. Modification: Added float32 normalization checks, in-place L2 normalization, and disk caching."),
        ("Feature 2: Code-Aware BM25 Lexical Retriever", "Both", "AI Tool: Antigravity AI. Prompt: 'Build code-aware BM25 tokenizer for camelCase and snake_case'. Output Summary: Built regex-based tokenizer and BM25Okapi ranker. Modification: Refined regex patterns to avoid multiline string consumption and handled fallback tokenizer."),
        ("Feature 3: Reciprocal Rank Fusion (RRF) Hybrid Retriever", "Both", "AI Tool: Antigravity AI. Prompt: 'Implement RRF fusion combining dense and lexical ranks'. Output Summary: Built HybridRetriever using standard RRF formula 1/(k + rank). Modification: Added configurable weighting parameter between semantic and lexical scores."),
        ("Feature 4: Version-Aware Retrieval Engine", "Both", "AI Tool: Antigravity AI. Prompt: 'Create version-aware index manager and dataset versioner'. Output Summary: Implemented IndexVersioner and DatasetVersioner. Modification: Added metadata persistence, version tagging, and fallback logic for missing version tags."),
        ("Feature 5: MTEB AppsRetrieval Evaluation Wrapper", "Both", "AI Tool: Antigravity AI. Prompt: 'Wrap code retrieval model for MTEB 2.x AppsRetrieval evaluation'. Output Summary: Created CodeRetrievalModel inheriting from SentenceTransformerEncoderWrapper. Modification: Added DataLoader unwrapping and Transformers 5.x compatibility patches.")
    ]

    for fname, classification, desc in features:
        p1 = doc.add_paragraph()
        p1.add_run("1. Feature Name: ").bold = True
        p1.add_run(fname)
        p2 = doc.add_paragraph()
        p2.add_run("2. Self-Generated / AI-Generated / Both: ").bold = True
        p2.add_run(classification)
        p3 = doc.add_paragraph()
        p3.add_run("3. Description (include AI tools/platform Used, prompt used, Output Summary, Modification): ").bold = True
        p3.add_run(desc)
        doc.add_paragraph() # Spacer

    # 5. Ethical & Compliance Confirmation
    doc.add_heading("5. Ethical & Compliance Confirmation", level=2)
    p = doc.add_paragraph()
    p.add_run("AI usage complies with guidelines and policies. ").bold = True
    p.add_run("Yes\n")
    p.add_run("No proprietary or copyrighted data misused. ").bold = True
    p.add_run("I Agree")

    # 6. Declaration & Sign-Off
    doc.add_heading("6. Declaration & Sign-Off", level=2)
    p = doc.add_paragraph()
    p.add_run("Name of Team Representative: ").bold = True
    p.add_run("_______________________________\n")
    p.add_run("Role: ").bold = True
    p.add_run("Team Lead\n")
    p.add_run("Signature: ").bold = True
    p.add_run("_______________________________\n")
    p.add_run("Date: ").bold = True
    p.add_run("September 27, 2026")

    output_path = "c:\\Users\\adhar\\Desktop\\samsung\\AI_Usage_Disclosure_Form.docx"
    doc.save(output_path)
    print(f"Disclosure form saved to: {output_path}")


if __name__ == "__main__":
    build_pptx()
    build_docx()
