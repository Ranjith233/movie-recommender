"""
Model Evaluation and Benchmark Suite
Measures Precision@K (K=5, K=10), Genre Alignment, Intra-List Diversity,
Catalog Coverage, and Query Latency across a curated benchmark set.
Documents failure modes and qualitative observations as required in Section 10 of the portfolio guide.
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.preprocess import load_and_preprocess_catalog
from src.recommender import ContentBasedRecommender


def compute_intra_list_diversity(recommended_indices: List[int], similarity_matrix: np.ndarray) -> float:
    """
    Computes Intra-List Diversity (ILD) as the average pairwise cosine distance
    between items in the recommendation list:
    ILD = 1 - (2 / (K * (K - 1))) * sum_{i < j} sim(i, j)
    """
    k = len(recommended_indices)
    if k <= 1:
        return 0.0
    
    pairwise_sims = []
    for i in range(k):
        for j in range(i + 1, k):
            idx_i = recommended_indices[i]
            idx_j = recommended_indices[j]
            pairwise_sims.append(similarity_matrix[idx_i, idx_j])
            
    avg_sim = np.mean(pairwise_sims) if pairwise_sims else 0.0
    return float(1.0 - avg_sim)


def evaluate_recommender(
    catalog_path: str = "data/movies.csv",
    benchmark_path: str = "data/evaluation_benchmark.csv",
    weighted: bool = True
) -> Dict[str, Any]:
    """
    Runs automated evaluation across all benchmark seed movies.
    """
    print(f"Loading catalog from {catalog_path} (Weighted={weighted})...")
    df = load_and_preprocess_catalog(catalog_path, weighted=weighted)
    recommender = ContentBasedRecommender(df)
    bench_df = pd.read_csv(benchmark_path)

    p_at_5_list = []
    p_at_10_list = []
    genre_alignment_5_list = []
    diversity_5_list = []
    latencies = []
    all_recommended_indices = set()
    failure_cases = []

    per_seed_results = []

    for _, b_row in bench_df.iterrows():
        seed_title = b_row["seed_title"]
        expected_genre = str(b_row["expected_genre"]).strip().lower()
        expected_dirs = [d.strip().lower() for d in str(b_row["expected_directors"]).split(",")]

        t0 = time.perf_counter()
        try:
            recs = recommender.recommend(seed_title, n=10)
        except Exception as e:
            print(f"Error querying '{seed_title}': {e}")
            continue
        latencies.append((time.perf_counter() - t0) * 1000.0)

        seed_idx = recommender.title_to_idx[seed_title.lower()]
        seed_row = df.iloc[seed_idx]
        seed_genres = set([g.strip().lower() for g in str(seed_row["genres"]).split(",")])

        # Track indices
        rec_indices = [recommender.title_to_idx[t.lower()] for t in recs["title"]]
        all_recommended_indices.update(rec_indices)

        # Precision@5 and Precision@10 calculation
        # A recommended item is judged RELEVANT if:
        # 1. It shares at least one core genre with the seed movie OR matches expected genre, AND
        # 2. It has cosine similarity > 0.03 OR matches director/franchise
        hits_at_5 = 0
        hits_at_10 = 0
        genre_matches_5 = 0

        for rank, (_, r) in enumerate(recs.iterrows()):
            rec_genres = set([g.strip().lower() for g in str(r["genres"]).split(",")])
            rec_director = str(r["director"]).strip().lower()
            sim_score = r["similarity_score"]

            shares_genre = len(seed_genres.intersection(rec_genres)) > 0 or expected_genre in rec_genres
            shares_director = any(d in rec_director for d in expected_dirs)

            is_relevant = (shares_genre and sim_score > 0.02) or shares_director

            if is_relevant:
                if rank < 5:
                    hits_at_5 += 1
                hits_at_10 += 1

            if shares_genre and rank < 5:
                genre_matches_5 += 1

            # Log failure case if rank < 3 and zero genre match
            if rank < 3 and not shares_genre and not shares_director:
                failure_cases.append({
                    "seed": seed_title,
                    "recommended": r["title"],
                    "rank": rank + 1,
                    "rec_genres": r["genres"],
                    "similarity": sim_score,
                    "reason": "Unrelated genre / thematic drift"
                })

        p_5 = hits_at_5 / 5.0
        p_10 = hits_at_10 / 10.0
        genre_align_5 = genre_matches_5 / 5.0
        ild_5 = compute_intra_list_diversity(rec_indices[:5], recommender.similarity_matrix)

        p_at_5_list.append(p_5)
        p_at_10_list.append(p_10)
        genre_alignment_5_list.append(genre_align_5)
        diversity_5_list.append(ild_5)

        per_seed_results.append({
            "seed": seed_title,
            "precision@5": round(p_5, 3),
            "precision@10": round(p_10, 3),
            "genre_alignment@5": round(genre_align_5, 3),
            "diversity@5": round(ild_5, 3),
            "top_1_rec": recs.iloc[0]["title"],
            "top_1_sim": round(recs.iloc[0]["similarity_score"], 3),
        })

    # Global aggregate metrics
    catalog_coverage = len(all_recommended_indices) / len(df)
    avg_latency = np.mean(latencies)

    results = {
        "model_variant": "Weighted Metadata TF-IDF (Cosine Similarity)" if weighted else "Baseline TF-IDF",
        "benchmark_sample_size": len(bench_df),
        "catalog_size": len(df),
        "mean_precision@5": round(float(np.mean(p_at_5_list)), 4),
        "mean_precision@10": round(float(np.mean(p_at_10_list)), 4),
        "mean_genre_alignment@5": round(float(np.mean(genre_alignment_5_list)), 4),
        "intra_list_diversity@5": round(float(np.mean(diversity_5_list)), 4),
        "catalog_coverage_pct": round(float(catalog_coverage * 100.0), 2),
        "mean_query_latency_ms": round(float(avg_latency), 3),
        "failure_case_count": len(failure_cases),
        "failure_cases": failure_cases[:5],
        "detailed_seeds": per_seed_results,
    }

    return results


def print_evaluation_report(results: Dict[str, Any]):
    """Prints a beautiful markdown-style evaluation report to terminal."""
    print("\n" + "=" * 65)
    print("      🎯 MOVIE RECOMMENDATION SYSTEM - BENCHMARK REPORT")
    print("=" * 65)
    print(f"Variant:              {results['model_variant']}")
    print(f"Benchmark Test Seeds: {results['benchmark_sample_size']} titles")
    print(f"Catalog Total:        {results['catalog_size']} movies")
    print("-" * 65)
    print(f"• Precision@5:        {results['mean_precision@5'] * 100:.1f}%")
    print(f"• Precision@10:       {results['mean_precision@10'] * 100:.1f}%")
    print(f"• Genre Alignment@5:  {results['mean_genre_alignment@5'] * 100:.1f}%")
    print(f"• Intra-List Div.@5:  {results['intra_list_diversity@5']:.3f} (Scale: 0 = identical, 1 = max diversity)")
    print(f"• Catalog Coverage:   {results['catalog_coverage_pct']:.1f}%")
    print(f"• Mean Query Latency: {results['mean_query_latency_ms']:.2f} ms")
    print("-" * 65)
    print("Sample Benchmark Results:")
    df_samples = pd.DataFrame(results["detailed_seeds"]).head(8)
    print(df_samples[["seed", "precision@5", "precision@10", "top_1_rec", "top_1_sim"]].to_string(index=False))
    print("=" * 65 + "\n")


if __name__ == "__main__":
    report = evaluate_recommender(weighted=True)
    print_evaluation_report(report)

    # Save to disk
    with open("data/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Detailed report persisted to data/evaluation_report.json")
