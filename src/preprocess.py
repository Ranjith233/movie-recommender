import re
import pandas as pd
from typing import List, Optional

TEXT_COLUMNS = ["genres", "overview", "keywords", "cast", "director"]

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def collapse_names(names_str: str) -> str:
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
    tags_list = []
    for _, row in df.iterrows():
        genres = (" " + clean_text(str(row.get("genres", "")))) * genre_weight
        keywords = (" " + clean_text(str(row.get("keywords", "")))) * keyword_weight
        overview = " " + clean_text(str(row.get("overview", "")))
        
        director_raw = clean_text(str(row.get("director", "")))
        director_entity = collapse_names(str(row.get("director", "")))
        director_block = (" " + director_raw + " " + director_entity) * director_weight
        
        cast_raw = clean_text(str(row.get("cast", "")))
        cast_entity = collapse_names(str(row.get("cast", "")))
        cast_block = (" " + cast_raw + " " + cast_entity) * cast_weight
        
        combined_doc = f"{genres}{keywords}{overview}{director_block}{cast_block}"
        combined_doc = re.sub(r"\s+", " ", combined_doc).strip()
        tags_list.append(combined_doc)
    return pd.Series(tags_list, index=df.index)

def load_and_preprocess_catalog(filepath: str, weighted: bool = True) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    required = ["title", "genres", "overview"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {filepath}")
            
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

    if weighted:
        df["tags"] = build_feature_tags(df, genre_weight=3, keyword_weight=2, director_weight=2)
    else:
        df["tags"] = (
            df["genres"].str.lower() + " " +
            df["keywords"].str.lower() + " " +
            df["overview"].str.lower() + " " +
            df["director"].str.lower()
        ).fillna("")

    return df
