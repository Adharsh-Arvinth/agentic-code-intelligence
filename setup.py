from setuptools import setup, find_packages

setup(
    name="agentic-code-intelligence",
    version="1.0.0",
    description="Agentic Code Intelligence: Multi-Stage Code Retrieval and Execution-Guided Reranking Engine",
    author="Adharsh Arvinth, Rahul J, Sivamiruthula",
    author_email="adharsharvinth2108@gmail.com",
    packages=find_packages(),
    py_modules=["agentic_code_intelligence"],
    install_requires=[
        "torch>=2.0.0",
        "sentence-transformers>=3.0.0",
        "transformers>=4.40.0",
        "faiss-cpu>=1.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "datasets>=2.19.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "rank-bm25>=0.2.4",
    ],
    python_requires=">=3.10",
)
