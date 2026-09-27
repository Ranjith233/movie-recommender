"""
Data Preprocessing & Feature Engineering Module
Part of the Content-Based Movie Recommendation System

Handles data ingestion, missing value imputation, string normalization,
and multi-field text tag construction with customizable feature weighting.
"""

import re
import pandas as pd
from typing import List, Optional


TEXT_COLUMNS = ["genres", "overview", "keywords", "cast", "director"]


def clean_text(text: str) -> str:
    """
    Standardizes text by lowercasing, stripping special characters,
    and removing excessive whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase and strip
    text = text.lower().strip()
    # Remove special punctuation while keeping hyphens inside compound words
    text = re.sub(r"[^\w\s\-]", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def collapse_names(names_str: str) -> str:
    """
    Converts multi-word person names into single entities (e.g., 'Christopher Nolan' -> 'ChristopherNolan')
    to prevent vectorizers from mistakenly correlating unrelated people sharing first names.
    Also retains original tokens for soft matching.
    """
    if not isinstance(names_str, str) or not names_str.strip():
        return ""
    
    entities = [e.strip() for e in names_str.split(",") if e.strip()]
    collapsed = [re.sub(r"\s+", "", e.lower()) for e in entities]
    return " ".join(collapsed)


def build_feature_tags(
    df: pd.DataFrame,
    genre_weight: int = 3,
    keyword_weight: int = 2,
    director_weight: int = 2,
    cast_weight: int = 1,
) -> pd.Series:
    """
    Constructs a weighted composite 'tags' document for each movie.
    
    Why weighting matters:
    Movie overviews often contain long, incidental plot words that can dilute
    critical metadata signals. By repeating high-value deliberate metadata
    (genres, keywords, director) by an integer weighting factor, we increase their
    term frequencies (TF) in TF-IDF, ensuring that recommendations match core themes
    and creative styles rather than superficial vocabulary overlap.
    """
    tags_list = []
    
    for _, row in df.iterrows():
        # Cleaned individual fields
        genres = (" " + clean_text(str(row.get("genres", "")))) * genre_weight
        keywords = (" " + clean_text(str(row.get("keywords", "")))) * keyword_weight
        overview = " " + clean_text(str(row.get("overview", "")))
        
        # Entity collapsed and raw director/cast
        director_raw = clean_text(str(row.get("director", "")))
        director_entity = collapse_names(str(row.get("director", "")))
        director_block = (" " + director_raw + " " + director_entity) * director_weight
        
        cast_raw = clean_text(str(row.get("cast", "")))
        cast_entity = collapse_names(str(row.get("cast", "")))
        cast_block = (" " + cast_raw + " " + cast_entity) * cast_weight
        
        combined_doc = f"{genres}{keywords}{overview}{director_block}{cast_block}"
        # Normalize whitespace
        combined_doc = re.sub(r"\s+", " ", combined_doc).strip()
        tags_list.append(combined_doc)
        
    return pd.Series(tags_list, index=df.index)


def load_and_preprocess_catalog(filepath: str, weighted: bool = True) -> pd.DataFrame:
    """
    Loads raw movie CSV data, validates required schema, imputes missing values,
    and returns a clean DataFrame with computed 'tags'.
    """
    df = pd.read_csv(filepath)
    
    # 1. Validate required columns
    required = ["title", "genres", "overview"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {filepath}")
            
    # 2. Impute nulls for metadata
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
        else:
            df[col] = ""
            
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(7.0)
    else:
        df["rating"] = 7.0
        
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(2000).astype(int)
    else:
        df["release_year"] = 2000
        
    if "language" not in df.columns:
        df["language"] = "English"
    else:
        df["language"] = df["language"].fillna("English").astype(str).str.strip()
        
    if "popularity" not in df.columns:
        df["popularity"] = 75.0
    else:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(75.0)

    # 3. Build tags (weighted or unweighted baseline)
    if weighted:
        df["tags"] = build_feature_tags(df, genre_weight=3, keyword_weight=2, director_weight=2)
    else:
        # Standard unweighted baseline as outlined in section 4 of the guide
        df["tags"] = (
            df["genres"].str.lower() + " " +
            df["keywords"].str.lower() + " " +
            df["overview"].str.lower() + " " +
            df["director"].str.lower()
        ).fillna("")

    return df
