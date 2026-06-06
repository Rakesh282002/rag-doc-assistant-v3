"""
RAG Evaluation Script using RAGAS framework.

Evaluates the RAG pipeline on 3 key dimensions:
  1. Retrieval Quality — context precision, context recall
  2. Generation Quality — faithfulness, answer relevancy
  3. End-to-End — answer correctness, answer similarity

Usage:
    python -m evaluation.evaluate

    # With custom test set:
    python -m evaluation.evaluate --test-set evaluation/test_set.json

Prerequisites:
    pip install ragas datasets
"""

import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import GOOGLE_API_KEY
from backend.rag_pipeline import (
    query_documents,
    _load_faiss,
    _load_chunks,
    _embeddings,
    _cross_encoder,
    INITIAL_RETRIEVAL_K,
    MAX_RETRIEVAL_DOCS,
    SCORE_GAP,
)
from backend.semantic_cache import get_cache


# ---------------------------------------------------------------------------
# Test Set — Ground truth Q&A pairs
# Replace these with questions specific to YOUR uploaded documents.
# ---------------------------------------------------------------------------

DEFAULT_TEST_SET = [
{
"question": "What is the candidate's name?",
"ground_truth": "Nakka Rakesh",
},
{
"question": "What is his current role?",
"ground_truth": "Associate Software Engineer In Test at HealthEdge Software Pvt Ltd",
},
{
"question": "Where does he work?",
"ground_truth": "HealthEdge Software Pvt Ltd",
},
{
"question": "What is his educational qualification?",
"ground_truth": "Bachelor of Technology in Information Technology (Honors)",
},
{
"question": "Which college did he attend?",
"ground_truth": "S.R.K.R. Engineering College",
},
{
"question": "What is his CGPA in B.Tech?",
"ground_truth": "9.07",
},
{
"question": "What are his primary programming skills?",
"ground_truth": "Python, Java, C",
},
{
"question": "What tools does he use for performance testing?",
"ground_truth": "Apache JMeter, Postman",
},
{
"question": "What monitoring tools has he worked with?",
"ground_truth": "Datadog, Graylog",
},
{
"question": "What is one of his key achievements?",
"ground_truth": "Star Performer (Q3 2025) at HealthEdge",
},
{
"question": "What certifications does he have?",
"ground_truth": "Data Science and AI Campus Program, Data Analytics with Python (NPTEL), Data Science for Engineers (NPTEL), Programming Essentials in Python (Cisco)",
},
{
"question": "What project did he build for healthcare prediction?",
"ground_truth": "Heart Disease Prediction System using machine learning",
},

# ---- Additional Questions ----

{
    "question": "What internship experience does he have?",
    "ground_truth": "AI-ML Intern at Henotic Technology Pvt Ltd",
},
{
    "question": "What did he achieve during his internship?",
    "ground_truth": "Built ML models, improved performance using accuracy, precision, recall, and performed data preprocessing and feature selection",
},
{
    "question": "What technologies does he use in his current job?",
    "ground_truth": "Apache JMeter, Python, Datadog, Graylog, Excel, MySQL, Postman",
},
{
    "question": "How did he reduce manual effort in reporting?",
    "ground_truth": "By developing Python automation scripts for performance reporting, reducing effort by 70%",
},
{
    "question": "How did he improve SLA monitoring?",
    "ground_truth": "Implemented real-time SLA monitoring in Datadog with automated alerts, reducing manual effort by 80–90%",
},
{
    "question": "What is his experience in machine learning?",
    "ground_truth": "Built ML models using algorithms like Logistic Regression, Random Forest, SVM, KNN, and XGBoost",
},
{
    "question": "Which model performed best in his heart disease project?",
    "ground_truth": "XGBoost with 83.26% accuracy and 0.82 F1-score",
},
{
    "question": "What NLP techniques did he use in spam detection?",
    "ground_truth": "Stop word removal, lemmatization, and TF-IDF vectorization",
},
{
    "question": "What deep learning technique did he use for emotion recognition?",
    "ground_truth": "CNN with MFCC feature extraction",
},
{
    "question": "What is his experience with databases?",
    "ground_truth": "MySQL (Intermediate)",
},
{
    "question": "What awards has he received in college?",
    "ground_truth": "1st place in Techthon 2022 and 2nd place in Code Unscramble at Techfleet 2023",
},
{
    "question": "What web technologies does he know?",
    "ground_truth": "HTML, CSS, JavaScript, Flask",
},


]



# ---------------------------------------------------------------------------
# Evaluation Metrics (standalone, no RAGAS dependency required)
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def answer_similarity(generated: str, expected: str, embeddings) -> float:
    """Semantic similarity between generated answer and ground truth."""
    gen_emb = np.array(embeddings.embed_query(generated))
    exp_emb = np.array(embeddings.embed_query(expected))
    return cosine_similarity(gen_emb, exp_emb)


def answer_contains_ground_truth(generated: str, ground_truth: str) -> float:
    """
    Simple token-overlap score: what fraction of ground truth tokens
    appear in the generated answer (case-insensitive).
    """
    gt_tokens = set(ground_truth.lower().split())
    gen_tokens = set(generated.lower().split())
    if not gt_tokens:
        return 1.0
    overlap = gt_tokens & gen_tokens
    return len(overlap) / len(gt_tokens)


def faithfulness_check(answer: str, sources: list) -> float:
    """
    Simplified faithfulness: what fraction of answer sentences
    have supporting evidence in the source content.
    """
    if not sources:
        return 0.0

    # Combine all source content
    context = " ".join(s.get("content", "") for s in sources).lower()

    # Split answer into sentences
    sentences = [s.strip() for s in answer.replace(".", ".\n").split("\n") if s.strip()]
    if not sentences:
        return 1.0

    supported = 0
    for sent in sentences:
        # Check if key words from the sentence appear in context
        words = set(sent.lower().split()) - {"the", "a", "an", "is", "are", "was", "were", "his", "her", "he", "she", "it", "in", "on", "at", "to", "for", "of", "and", "or"}
        if not words:
            supported += 1
            continue
        matches = sum(1 for w in words if w in context)
        if matches / len(words) >= 0.5:
            supported += 1

    return supported / len(sentences)


def retrieval_hit_rate(question: str, ground_truth: str, sources: list) -> float:
    """Did any retrieved source contain ground truth keywords?"""
    if not sources:
        return 0.0
    gt_tokens = set(ground_truth.lower().split())
    for src in sources:
        content_tokens = set(src.get("content", "").lower().split())
        if gt_tokens & content_tokens:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# RAGAS-based evaluation (if installed)
# ---------------------------------------------------------------------------

def evaluate_with_ragas(results: list) -> dict | None:
    """
    Run RAGAS evaluation if the package is installed.
    Returns metric scores or None if RAGAS is unavailable.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        # Build RAGAS dataset
        data = {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [[s["content"] for s in r["sources"]] for r in results],
            "ground_truth": [r["ground_truth"] for r in results],
        }

        dataset = Dataset.from_dict(data)
        score = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return score.to_pandas().to_dict()
    except ImportError:
        return None
    except Exception as e:
        print(f"[RAGAS] Evaluation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(test_set: list, verbose: bool = True) -> dict:
    """
    Run the full evaluation pipeline.
    Returns a summary dict with per-question and aggregate metrics.
    """
    embeddings = _embeddings()
    results = []

    print(f"\n{'='*70}")
    print(f"  RAG EVALUATION — {len(test_set)} questions")
    print(f"{'='*70}\n")

    for i, item in enumerate(test_set, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i}/{len(test_set)}] {question}")
        t0 = time.perf_counter()

        # Run the RAG pipeline
        try:
            response = query_documents(question)
            answer = response["answer"]
            sources = response["sources"]
            is_cached = response.get("cached", False)
            latency = time.perf_counter() - t0
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": f"ERROR: {e}",
                "sources": [],
                "metrics": {},
                "latency": time.perf_counter() - t0,
            })
            continue

        # Compute metrics
        sim = answer_similarity(answer, ground_truth, embeddings)
        correctness = answer_contains_ground_truth(answer, ground_truth)
        faith = faithfulness_check(answer, sources)
        hit_rate = retrieval_hit_rate(question, ground_truth, sources)

        metrics = {
            "answer_similarity": round(sim, 4),
            "answer_correctness": round(correctness, 4),
            "faithfulness": round(faith, 4),
            "retrieval_hit_rate": round(hit_rate, 4),
            "latency_s": round(latency, 2),
            "cached": is_cached,
        }

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "sources": sources,
            "metrics": metrics,
            "latency": latency,
        })

        if verbose:
            print(f"  Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"  Similarity: {sim:.4f} | Correctness: {correctness:.4f} | "
                  f"Faithfulness: {faith:.4f} | Hit Rate: {hit_rate:.1f} | "
                  f"Latency: {latency:.2f}s | Cached: {is_cached}")
            print()

    # --- Aggregate metrics ---
    valid = [r for r in results if r["metrics"]]
    if not valid:
        print("No valid results to aggregate.")
        return {"results": results, "aggregate": {}}

    aggregate = {
        "avg_answer_similarity": round(np.mean([r["metrics"]["answer_similarity"] for r in valid]), 4),
        "avg_answer_correctness": round(np.mean([r["metrics"]["answer_correctness"] for r in valid]), 4),
        "avg_faithfulness": round(np.mean([r["metrics"]["faithfulness"] for r in valid]), 4),
        "avg_retrieval_hit_rate": round(np.mean([r["metrics"]["retrieval_hit_rate"] for r in valid]), 4),
        "avg_latency_s": round(np.mean([r["metrics"]["latency_s"] for r in valid]), 2),
        "cache_hit_rate": round(sum(1 for r in valid if r["metrics"]["cached"]) / len(valid), 4),
        "total_questions": len(test_set),
        "successful": len(valid),
    }

    print(f"\n{'='*70}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"  Answer Similarity  : {aggregate['avg_answer_similarity']:.4f}")
    print(f"  Answer Correctness : {aggregate['avg_answer_correctness']:.4f}")
    print(f"  Faithfulness       : {aggregate['avg_faithfulness']:.4f}")
    print(f"  Retrieval Hit Rate : {aggregate['avg_retrieval_hit_rate']:.4f}")
    print(f"  Avg Latency        : {aggregate['avg_latency_s']:.2f}s")
    print(f"  Cache Hit Rate     : {aggregate['cache_hit_rate']:.4f}")
    print(f"  Questions          : {aggregate['successful']}/{aggregate['total_questions']}")
    print(f"{'='*70}\n")

    # Try RAGAS evaluation
    ragas_scores = evaluate_with_ragas(results)
    if ragas_scores:
        print("  RAGAS Scores:")
        for k, v in ragas_scores.items():
            print(f"    {k}: {v}")
        aggregate["ragas"] = ragas_scores

    return {"results": results, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument(
        "--test-set", "-t",
        type=str,
        default=None,
        help="Path to JSON file with test set (list of {question, ground_truth})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="evaluation/results.json",
        help="Output path for detailed results JSON",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-question output",
    )
    args = parser.parse_args()

    # Load test set
    if args.test_set:
        with open(args.test_set, "r") as f:
            test_set = json.load(f)
    else:
        test_set = DEFAULT_TEST_SET
        print("[INFO] Using default test set. Create evaluation/test_set.json for custom questions.\n")

    # Run evaluation
    results = run_evaluation(test_set, verbose=not args.quiet)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[INFO] Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
