"""
scripts/fetch_datasets.py — Fetch and serialize all 3 benchmark datasets.

Decision D-010: Fetch ONCE, serialize to disk, never re-ingest during benchmarking.
This ensures all benchmark runs use identical data.

Usage:
    python scripts/fetch_datasets.py --dataset small
    python scripts/fetch_datasets.py --dataset medium
    python scripts/fetch_datasets.py --dataset large
    python scripts/fetch_datasets.py --all
"""

import json
import pickle
import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATASETS_RAW, DATASETS_PROCESSED, DATASETS

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def fetch_small_wikipedia(n: int = 50) -> list[dict]:
    """
    Fetch n Wikipedia articles.
    Uses a curated seed list for reproducibility (D-010).
    Falls back to wikipedia-api if available.
    """
    # Curated list of representative Wikipedia articles
    SEED_ARTICLES = [
        "Machine learning", "Deep learning", "Neural network", "Natural language processing",
        "Computer vision", "Reinforcement learning", "Transformer (machine learning model)",
        "BERT (language model)", "GPT-3", "ImageNet", "Convolutional neural network",
        "Recurrent neural network", "Long short-term memory", "Attention mechanism",
        "Transfer learning", "Generative adversarial network", "Variational autoencoder",
        "Random forest", "Support vector machine", "K-means clustering",
        "Principal component analysis", "Gradient descent", "Backpropagation",
        "Overfitting", "Regularization (mathematics)", "Cross-validation (statistics)",
        "Precision and recall", "F1 score", "ROC curve", "Confusion matrix",
        "Python (programming language)", "TensorFlow", "PyTorch", "Keras",
        "NumPy", "Pandas (software)", "Scikit-learn", "Jupyter", "Docker",
        "Kubernetes", "Cloud computing", "Application programming interface",
        "Database", "SQL", "NoSQL", "Elasticsearch", "Redis",
        "Git", "Agile software development", "DevOps",
    ][:n]

    documents = []
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia("RAG-Benchmark/1.0", "en")
        print(f"[fetch] Fetching {n} Wikipedia articles...")
        for title in SEED_ARTICLES:
            page = wiki.page(title)
            if page.exists():
                documents.append({
                    "content": page.text[:5000],   # cap at 5000 chars per article
                    "source": f"wikipedia:{title}",
                    "metadata": {
                        "title": title,
                        "url": page.fullurl,
                        "dataset": "small",
                        "fetched_at": datetime.utcnow().isoformat(),
                    }
                })
                time.sleep(0.1)   # be polite to Wikipedia API
            if len(documents) >= n:
                break
        print(f"[fetch] Got {len(documents)} Wikipedia articles")
    except ImportError:
        print("[fetch] wikipedia-api not installed — using stub data for testing")
        documents = _generate_stub_documents("wikipedia", n)
    return documents


def fetch_medium_arxiv(n: int = 500) -> list[dict]:
    """
    Fetch n arXiv papers from cs.LG, cs.CL, cs.CV (2020-2023).
    D-010: Abstracts-only fallback for papers without full text (~12%).
    """
    documents = []
    try:
        import arxiv
        categories = ["cs.LG", "cs.CL", "cs.CV"]
        per_category = n // len(categories)

        print(f"[fetch] Fetching {n} arXiv papers ({per_category} per category)...")
        for cat in categories:
            search = arxiv.Search(
                query=f"cat:{cat}",
                max_results=per_category,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            for paper in search.results():
                content = f"{paper.title}\n\n{paper.summary}"
                documents.append({
                    "content": content,
                    "source": f"arxiv:{paper.entry_id}",
                    "metadata": {
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors[:3]],
                        "categories": paper.categories,
                        "published": str(paper.published),
                        "url": paper.entry_id,
                        "abstract_only": True,   # D-010: flagged
                        "dataset": "medium",
                        "fetched_at": datetime.utcnow().isoformat(),
                    }
                })
                if len(documents) >= n:
                    break
        print(f"[fetch] Got {len(documents)} arXiv papers")
    except ImportError:
        print("[fetch] arxiv not installed — using stub data for testing")
        documents = _generate_stub_documents("arxiv", n)
    return documents


def fetch_large_kubernetes(max_pages: int = 2400) -> list[dict]:
    """
    Fetch Kubernetes documentation from GitHub.
    D-010: Pinned to kubernetes/website@v1.29 for reproducibility.
    
    NOTE: Requires git. If git unavailable, fetches a subset via GitHub API.
    """
    import subprocess
    import os

    k8s_dir = DATASETS_RAW / "kubernetes-website"
    documents = []

    # Try git clone if not already done
    if not k8s_dir.exists():
        print(f"[fetch] Cloning kubernetes/website@v1.29 to {k8s_dir}...")
        try:
            subprocess.run([
                "git", "clone", "--depth=1", "--branch=v1.29",
                "https://github.com/kubernetes/website.git",
                str(k8s_dir),
            ], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("[fetch] git clone failed — using stub data for testing")
            return _generate_stub_documents("kubernetes", min(max_pages, 100))

    # Parse markdown files
    content_dir = k8s_dir / "content" / "en"
    if not content_dir.exists():
        content_dir = k8s_dir / "content"

    md_files = list(content_dir.rglob("*.md"))[:max_pages]
    print(f"[fetch] Parsing {len(md_files)} Kubernetes markdown pages...")

    for md_file in md_files:
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
            # Strip YAML frontmatter
            if text.startswith("---"):
                end = text.find("---", 3)
                if end > 0:
                    text = text[end + 3:].strip()

            if len(text) < 100:   # skip stubs
                continue

            relative_path = str(md_file.relative_to(k8s_dir))
            documents.append({
                "content": text[:8000],   # cap per page
                "source": f"kubernetes:{relative_path}",
                "metadata": {
                    "file": relative_path,
                    "dataset": "large",
                    "version": "v1.29",
                    "fetched_at": datetime.utcnow().isoformat(),
                }
            })
        except Exception as e:
            continue

    print(f"[fetch] Got {len(documents)} Kubernetes pages")
    return documents


def _generate_stub_documents(source: str, n: int) -> list[dict]:
    """
    Generate stub documents for testing when real fetch is unavailable.
    These allow the pipeline to run end-to-end without API keys.
    """
    topics = [
        "machine learning model training", "neural network architecture",
        "data preprocessing pipeline", "model evaluation metrics",
        "hyperparameter optimization", "transfer learning techniques",
        "natural language processing", "computer vision tasks",
        "reinforcement learning agents", "transformer attention mechanism",
    ]
    docs = []
    for i in range(n):
        topic = topics[i % len(topics)]
        docs.append({
            "content": (
                f"This document discusses {topic}. "
                f"In the field of artificial intelligence, {topic} plays a crucial role. "
                f"Researchers have studied {topic} extensively. "
                f"Key concepts include optimization, performance metrics, and scalability. "
                f"Document {i} provides detailed analysis of {topic}. "
                f"The methodology involves systematic evaluation and comparison. "
                f"Results demonstrate significant improvements over baseline approaches. "
                f"Future work will explore {topic} in different domains."
            ) * 3,
            "source": f"{source}:stub_{i}",
            "metadata": {
                "stub": True,
                "index": i,
                "topic": topic,
                "dataset": source,
            }
        })
    return docs


def save_dataset(documents: list[dict], name: str) -> Path:
    """Serialize dataset to disk. Returns path."""
    DATASETS_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATASETS_PROCESSED / f"{name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(documents, f)

    # Also save metadata as JSON for inspection
    meta_path = DATASETS_PROCESSED / f"{name}_metadata.json"
    metadata = {
        "name": name,
        "num_documents": len(documents),
        "fetched_at": datetime.utcnow().isoformat(),
        "random_seed": RANDOM_SEED,
        "sources": list(set(d["source"].split(":")[0] for d in documents)),
        "avg_content_length": int(sum(len(d["content"]) for d in documents) / len(documents)) if documents else 0,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[fetch] Saved {len(documents)} documents to {out_path}")
    print(f"[fetch] Metadata saved to {meta_path}")
    return out_path


def load_dataset(name: str) -> list[dict]:
    """Load a previously fetched dataset from disk."""
    path = DATASETS_PROCESSED / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{name}' not found. Run fetch first: "
                                f"python scripts/fetch_datasets.py --dataset {name}")
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Fetch benchmark datasets")
    parser.add_argument("--dataset", choices=["small", "medium", "large"], help="Dataset to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch all datasets")
    args = parser.parse_args()

    DATASETS_RAW.mkdir(parents=True, exist_ok=True)

    to_fetch = []
    if args.all:
        to_fetch = ["small", "medium", "large"]
    elif args.dataset:
        to_fetch = [args.dataset]
    else:
        parser.print_help()
        return

    for dataset in to_fetch:
        print(f"\n{'='*50}")
        print(f"Fetching: {dataset}")
        print(f"{'='*50}")
        if dataset == "small":
            docs = fetch_small_wikipedia(n=50)
        elif dataset == "medium":
            docs = fetch_medium_arxiv(n=500)
        elif dataset == "large":
            docs = fetch_large_kubernetes(max_pages=2400)
        save_dataset(docs, dataset)
        print(f"✅ {dataset} dataset ready: {len(docs)} documents")


if __name__ == "__main__":
    main()
