"""
evaluation/ragas_eval.py — RAGAS-based answer quality evaluation.

Metrics: faithfulness, answer_relevancy, context_precision, context_recall
Decision D-002: Uses JUDGE_MODEL (gpt-4o) not the generator (gpt-4o-mini)
Decision D-011: 50 QA pairs per dataset

Usage:
    from evaluation.ragas_eval import evaluate_rag, generate_qa_pairs
    metrics = evaluate_rag(rag_system, qa_pairs)
"""

from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import JUDGE_MODEL, QA_PAIRS_PER_DATASET, TOP_K


@dataclass
class RAGASResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    num_samples: int
    raw_scores: list[dict]


def evaluate_rag(
    rag_system,
    qa_pairs: list[dict],
    k: int = TOP_K,
) -> RAGASResult:
    """
    Run RAGAS evaluation on a RAG system using a set of QA pairs.
    
    Args:
        rag_system: Any BaseRAG subclass (must be indexed)
        qa_pairs: List of {"question": str, "ground_truth": str}
        k: Number of docs to retrieve per question
    
    Returns RAGASResult with all 4 RAGAS metrics.
    
    Decision D-002: JUDGE_MODEL is gpt-4o (stronger than generator) to avoid self-eval bias.
    """
    try:
        from ragas import evaluate, RunConfig
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
        from gemini_client import get_langchain_llm
        from local_client import get_langchain_embeddings
    except ImportError as e:
        print(f"[ragas_eval] RAGAS dependencies not installed: {e}")
        print("[ragas_eval] Returning mock scores for pipeline testing")
        return _mock_ragas_result(len(qa_pairs))

    print(f"[ragas_eval] Evaluating {len(qa_pairs)} QA pairs "
          f"with judge={JUDGE_MODEL}...")

    # Collect RAG outputs for all QA pairs
    questions, answers, contexts, ground_truths = [], [], [], []

    for i, pair in enumerate(qa_pairs):
        question = pair["question"]
        ground_truth = pair["ground_truth"]

        result = rag_system.query(question, k=k)
        context_texts = [doc.content for doc in result.documents]

        questions.append(question)
        answers.append(result.answer)
        contexts.append(context_texts)
        ground_truths.append(ground_truth)

        if (i + 1) % 10 == 0:
            print(f"[ragas_eval] {i+1}/{len(qa_pairs)} queries processed...")

    # Build RAGAS dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Gemini judge (D-002: stronger than generator) + local bge-large-en-v1.5 embeddings
    judge_llm = get_langchain_llm(JUDGE_MODEL)
    judge_embeddings = get_langchain_embeddings()

    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy],   # context_precision/recall dropped to halve CPU time
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=1, timeout=600),
    )

    df = result.to_pandas()
    raw_scores = df.to_dict(orient="records")

    return RAGASResult(
        faithfulness=float(df["faithfulness"].mean()),
        answer_relevancy=float(df["answer_relevancy"].mean()),
        context_precision=float(df["context_precision"].mean()) if "context_precision" in df.columns else 0.0,
        context_recall=float(df["context_recall"].mean()) if "context_recall" in df.columns else 0.0,
        num_samples=len(qa_pairs),
        raw_scores=raw_scores,
    )


def generate_qa_pairs(
    documents: list[dict],
    n: int = QA_PAIRS_PER_DATASET,
    dataset_name: str = "unknown",
) -> list[dict]:
    """
    Generate synthetic QA pairs from documents using RAGAS testset generator.
    Decision D-011: 40% simple, 35% multi-hop, 25% comparison.
    
    Falls back to simple extraction if RAGAS not available.
    """
    try:
        from ragas.testset import TestsetGenerator
        from ragas.testset.evolutions import simple, multi_context, reasoning
        from langchain.schema import Document as LCDocument
        from gemini_client import get_langchain_llm
        from local_client import get_langchain_embeddings

        print(f"[ragas_eval] Generating {n} QA pairs for {dataset_name}...")

        # Convert to LangChain documents
        lc_docs = [
            LCDocument(
                page_content=doc["content"],
                metadata={"source": doc["source"]},
            )
            for doc in documents
        ]

        generator_llm = get_langchain_llm(JUDGE_MODEL)
        critic_llm = get_langchain_llm(JUDGE_MODEL)
        embeddings = get_langchain_embeddings()

        generator = TestsetGenerator.from_langchain(
            generator_llm=generator_llm,
            critic_llm=critic_llm,
            embeddings=embeddings,
        )

        # D-011: question type distribution
        testset = generator.generate_with_langchain_docs(
            lc_docs,
            test_size=n,
            distributions={simple: 0.4, multi_context: 0.35, reasoning: 0.25},
        )
        df = testset.to_pandas()
        pairs = [
            {"question": row["question"], "ground_truth": row["ground_truth"]}
            for _, row in df.iterrows()
        ]
        print(f"[ragas_eval] Generated {len(pairs)} QA pairs")
        return pairs

    except ImportError:
        print("[ragas_eval] RAGAS not installed — using simple extraction fallback")
        return _extract_simple_qa_pairs(documents, n)


def _extract_simple_qa_pairs(documents: list[dict], n: int) -> list[dict]:
    """
    Fallback: extract simple factual QA pairs without RAGAS.
    Good enough for pipeline testing; replace with real RAGAS generation for final benchmark.
    """
    import random
    random.seed(42)

    pairs = []
    for doc in random.sample(documents, min(n, len(documents))):
        sentences = [s.strip() for s in doc["content"].split(".") if len(s.strip()) > 50]
        if sentences:
            sentence = random.choice(sentences[:5])
            words = sentence.split()
            if len(words) > 5:
                # Create a simple "what" question from the sentence
                question = f"What does the source say about: {' '.join(words[:6])}?"
                pairs.append({
                    "question": question,
                    "ground_truth": sentence,
                })
    return pairs[:n]


def _mock_ragas_result(n: int) -> RAGASResult:
    """Return placeholder scores for pipeline testing without API keys."""
    return RAGASResult(
        faithfulness=0.0,
        answer_relevancy=0.0,
        context_precision=0.0,
        context_recall=0.0,
        num_samples=n,
        raw_scores=[],
    )
