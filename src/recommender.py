import difflib
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(
        self,
        df: pd.DataFrame,
        stop_words: str = "english",
        max_features: int = 20000,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        self.df = df.reset_index(drop=True).copy()
        self.stop_words = stop_words
        self.max_features = max_features
        self.ngram_range = ngram_range

        self.title_to_idx = {
            str(title).strip().lower(): idx for idx, title in enumerate(self.df["title"])
        }
        self.all_titles = list(self.df["title"].values)

        self.vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["tags"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

    def find_title(self, query: str, cutoff: float = 0.5) -> Tuple[Optional[str], Optional[int], List[str]]:
        clean_q = query.strip().lower()
        if not clean_q:
            return None, None, []

        if clean_q in self.title_to_idx:
            idx = self.title_to_idx[clean_q]
            return self.df.iloc[idx]["title"], idx, []

        substring_matches = [
            t for t in self.all_titles if clean_q in t.lower()
        ]
        if len(substring_matches) == 1:
            title = substring_matches[0]
            return title, self.title_to_idx[title.lower()], []

        close_matches = difflib.get_close_matches(
            clean_q, [t.lower() for t in self.all_titles], n=5, cutoff=cutoff
        )
        if close_matches:
            best_lower = close_matches[0]
            best_idx = self.title_to_idx[best_lower]
            alternatives = [self.df.iloc[self.title_to_idx[m]]["title"] for m in close_matches[1:]]
            return self.df.iloc[best_idx]["title"], best_idx, alternatives

        return None, None, substring_matches[:5]

    def _extract_shared_terms(self, idx1: int, idx2: int, top_k: int = 5) -> List[str]:
        row1 = self.tfidf_matrix[idx1].toarray().flatten()
        row2 = self.tfidf_matrix[idx2].toarray().flatten()
        overlap = row1 * row2
        top_indices = np.argsort(overlap)[::-1]
        
        suppress_tokens = set()
        for idx in [idx1, idx2]:
            for field in ["director", "cast"]:
                val = str(self.df.iloc[idx].get(field, "")).lower()
                for token in val.replace(",", " ").split():
                    suppress_tokens.add(token.strip())
                    suppress_tokens.add(token.replace(" ", "").strip())

        shared = []
        for i in top_indices:
            if overlap[i] > 0.001:
                term = self.feature_names[i]
                clean_term = term.replace(" ", "")
                if clean_term in suppress_tokens or any(s in term for s in suppress_tokens if len(s) > 4):
                    continue
                if term not in shared and len(term) > 2:
                    shared.append(term)
                if len(shared) >= top_k:
                    break
        return shared

    def _generate_explanation(self, seed_row: pd.Series, target_row: pd.Series, shared_terms: List[str]) -> str:
        reasons = []

        if seed_row["director"] and target_row["director"]:
            d1 = set([d.strip().lower() for d in seed_row["director"].split(",")])
            d2 = set([d.strip().lower() for d in target_row["director"].split(",")])
            common_dir = d1.intersection(d2)
            if common_dir:
                reasons.append(f"Directed by {target_row['director']}")

        if seed_row["genres"] and target_row["genres"]:
            g1 = set([g.strip().lower() for g in seed_row["genres"].split(",")])
            g2 = set([g.strip().lower() for g in target_row["genres"].split(",")])
            common_genres = [g.title() for g in g1.intersection(g2)]
            if common_genres:
                reasons.append(f"Shared genres: {', '.join(common_genres)}")

        if seed_row["cast"] and target_row["cast"]:
            c1 = set([c.strip().lower() for c in seed_row["cast"].split(",")])
            c2 = set([c.strip().lower() for c in target_row["cast"].split(",")])
            common_cast = [c.title() for c in c1.intersection(c2)]
            if common_cast:
                reasons.append(f"Featuring {', '.join(common_cast[:2])}")

        thematic_terms = [t for t in shared_terms if not any(g.lower() in t for g in seed_row["genres"].split(","))]
        if thematic_terms:
            reasons.append(f"Thematic keywords: {', '.join(thematic_terms[:3])}")

        if not reasons:
            return "Strong thematic and narrative similarity."
        return " • ".join(reasons)

    def recommend(
        self,
        title: str,
        n: int = 10,
        genre_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_rating: Optional[float] = None,
        popularity_weight: float = 0.0,
        rating_weight: float = 0.15,
        apply_diversity: bool = False,
    ) -> pd.DataFrame:
        resolved_title, seed_idx, alternatives = self.find_title(title)
        if resolved_title is None or seed_idx is None:
            raise KeyError(
                f"Movie '{title}' was not found in catalog. "
                f"Did you mean: {', '.join(alternatives) if alternatives else 'N/A'}?"
            )

        seed_row = self.df.iloc[seed_idx]
        sim_scores = self.similarity_matrix[seed_idx].copy()

        content_weight = max(0.0, 1.0 - popularity_weight - rating_weight)
        norm_pop = self.df["popularity"] / 100.0
        norm_rating = self.df["rating"] / 10.0

        hybrid_scores = (
            (content_weight * sim_scores) +
            (popularity_weight * norm_pop.values) +
            (rating_weight * norm_rating.values)
        )
        hybrid_scores[seed_idx] = -1.0

        candidate_indices = np.argsort(hybrid_scores)[::-1]
        recommendations = []
        seen_franchise_directors = {}

        for idx in candidate_indices:
            if hybrid_scores[idx] <= 0:
                continue

            row = self.df.iloc[idx]

            if genre_filter and genre_filter.strip().lower() != "all":
                target_genres = [g.strip().lower() for g in str(row["genres"]).split(",")]
                if genre_filter.strip().lower() not in target_genres:
                    continue

            if language_filter and language_filter.strip().lower() != "all":
                if str(row["language"]).strip().lower() != language_filter.strip().lower():
                    continue

            if min_year is not None and row["release_year"] < min_year:
                continue
            if max_year is not None and row["release_year"] > max_year:
                continue

            if min_rating is not None and row["rating"] < min_rating:
                continue

            if apply_diversity:
                d_key = str(row["director"]).lower()
                count = seen_franchise_directors.get(d_key, 0)
                if count >= 2:
                    continue
                seen_franchise_directors[d_key] = count + 1

            shared_terms = self._extract_shared_terms(seed_idx, idx, top_k=5)
            explanation = self._generate_explanation(seed_row, row, shared_terms)

            rec_item = {
                "title": row["title"],
                "genres": row["genres"],
                "release_year": int(row["release_year"]),
                "rating": float(row["rating"]),
                "director": row["director"],
                "language": row["language"],
                "overview": row["overview"],
                "similarity_score": round(float(sim_scores[idx]), 4),
                "hybrid_score": round(float(hybrid_scores[idx]), 4),
                "why_recommended": explanation,
                "shared_keywords": ", ".join(shared_terms[:4]),
            }
            recommendations.append(rec_item)

            if len(recommendations) >= n:
                break

        return pd.DataFrame(recommendations)
