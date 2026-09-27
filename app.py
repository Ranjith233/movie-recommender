import os
import json
import streamlit as st
import pandas as pd
import numpy as np

from src.preprocess import load_and_preprocess_catalog
from src.recommender import ContentBasedRecommender

st.set_page_config(
    page_title="CineMatch AI | Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f9fafb;
        margin-bottom: 8px;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #9ca3af;
        margin-bottom: 0;
    }

    .seed-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 24px;
    }

    .rec-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .rec-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }

    .badge-sim {
        background-color: #238636;
        color: #ffffff;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
    }
    .badge-rating {
        background-color: #d97706;
        color: #ffffff;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
    }
    .badge-year {
        background-color: #374151;
        color: #e5e7eb;
        font-size: 0.82rem;
        padding: 4px 8px;
        border-radius: 6px;
        display: inline-block;
    }
    .badge-genre {
        background-color: #1f2937;
        color: #93c5fd;
        border: 1px solid #3b82f6;
        font-size: 0.78rem;
        padding: 2px 8px;
        border-radius: 6px;
        margin-right: 4px;
        display: inline-block;
    }

    .why-box {
        background: #0f1d32;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 10px 0;
        font-size: 0.88rem;
        color: #7dd3fc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner="Loading catalog...")
def get_catalog_data():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "movies.csv")
    df = load_and_preprocess_catalog(csv_path, weighted=True)
    return df

@st.cache_resource(show_spinner="Indexing TF-IDF vectors...")
def get_recommender(df):
    return ContentBasedRecommender(df)

@st.cache_data
def get_evaluation_report():
    report_path = os.path.join(os.path.dirname(__file__), "data", "evaluation_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return json.load(f)
    return None

df = get_catalog_data()
recommender = get_recommender(df)
eval_report = get_evaluation_report()

st.sidebar.image(
    "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=800&auto=format&fit=crop&q=60",
    use_container_width=True,
)
st.sidebar.title("🎛️ Discovery Controls")

sorted_titles = sorted(df["title"].unique())
default_seed = "Inception" if "Inception" in sorted_titles else sorted_titles[0]
selected_title = st.sidebar.selectbox(
    "Select Seed Movie:",
    sorted_titles,
    index=sorted_titles.index(default_seed),
    help="Pick a movie you love to generate content-based recommendations.",
)

top_n = st.sidebar.slider("Number of Recommendations (Top-N):", min_value=3, max_value=15, value=6)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Metadata Filters")

all_genres_set = set()
for g_str in df["genres"].dropna():
    for g in g_str.split(","):
        if g.strip():
            all_genres_set.add(g.strip())
genres_list = ["All"] + sorted(list(all_genres_set))
selected_genre = st.sidebar.selectbox("Filter by Genre:", genres_list, index=0)

all_languages = ["All"] + sorted([l for l in df["language"].dropna().unique() if l])
selected_language = st.sidebar.selectbox("Filter by Language:", all_languages, index=0)

min_c_year = int(df["release_year"].min())
max_c_year = int(df["release_year"].max())
year_range = st.sidebar.slider(
    "Release Year Range:",
    min_value=min_c_year,
    max_value=max_c_year,
    value=(min_c_year, max_c_year),
    step=1,
)

min_rating = st.sidebar.slider(
    "Minimum IMDb Rating:",
    min_value=5.0,
    max_value=9.5,
    value=6.5,
    step=0.1,
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Ranking Improvements")
apply_hybrid = st.sidebar.checkbox(
    "Enable Quality Score Boost",
    value=True,
    help="Incorporates catalog rating as a secondary signal (PDF Section 8).",
)
rating_weight = 0.15 if apply_hybrid else 0.0

apply_diversity = st.sidebar.checkbox(
    "Apply Intra-List Diversity",
    value=True,
    help="Penalizes near-duplicate items from the same franchise or creator (PDF Section 8).",
)

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🎬 CineMatch AI: Explainable Recommendation Engine</div>
        <p class="hero-sub">
            Built with <strong>Python, Pandas, Scikit-Learn TF-IDF & Cosine Similarity</strong>. 
            Features multi-attribute feature weighting, real-time metadata filtering, and transparent reasoning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

seed_matches = df[df["title"] == selected_title]
if not seed_matches.empty:
    seed_row = seed_matches.iloc[0]
    st.markdown(
        f"""
        <div class="seed-box">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                <h3 style="margin: 0; color: #60a5fa;">🎯 Currently Analyzing: {seed_row['title']}</h3>
                <div>
                    <span class="badge-year">{int(seed_row['release_year'])}</span>
                    <span class="badge-rating">★ {seed_row['rating']:.1f}</span>
                    <span class="badge-year">{seed_row['language']}</span>
                </div>
            </div>
            <p style="margin: 4px 0 10px 0; color: #94a3b8; font-size: 0.95rem;">
                <strong>Director:</strong> {seed_row['director']} &nbsp;|&nbsp; 
                <strong>Cast:</strong> {seed_row['cast']}
            </p>
            <div style="margin-bottom: 8px;">
                {"".join([f"<span class='badge-genre'>{g.strip()}</span>" for g in str(seed_row['genres']).split(',')])}
            </div>
            <p style="margin: 0; font-size: 0.92rem; color: #cbd5e1; font-style: italic;">
                "{seed_row['overview']}"
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_recs, tab_analytics, tab_eval, tab_interview = st.tabs([
    "🍿 Recommendations Grid",
    "📊 Similarity & Term Breakdown",
    "🎯 Model Evaluation & Precision",
    "💡 Architecture & Interview Prep",
])

with tab_recs:
    try:
        recs_df = recommender.recommend(
            title=selected_title,
            n=top_n,
            genre_filter=selected_genre if selected_genre != "All" else None,
            language_filter=selected_language if selected_language != "All" else None,
            min_year=year_range[0],
            max_year=year_range[1],
            min_rating=min_rating,
            popularity_weight=0.05 if apply_hybrid else 0.0,
            rating_weight=rating_weight,
            apply_diversity=apply_diversity,
        )

        if recs_df.empty:
            st.warning("⚠️ No movies matched your specific filter combination. Try relaxing the year, genre, or minimum rating filters.")
        else:
            st.markdown(f"### Top {len(recs_df)} Personalized Matches for '{selected_title}'")
            col1, col2 = st.columns(2)
            for idx, (_, item) in enumerate(recs_df.iterrows()):
                target_col = col1 if idx % 2 == 0 else col2
                with target_col:
                    genre_pills = "".join(
                        [f"<span class='badge-genre'>{g.strip()}</span>" for g in str(item['genres']).split(',')]
                    )
                    sim_pct = int(item['similarity_score'] * 100)
                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <h4 style="margin: 0; color: #f3f4f6;">{idx + 1}. {item['title']}</h4>
                                <div>
                                    <span class="badge-year">{item['release_year']}</span>
                                    <span class="badge-rating">★ {item['rating']:.1f}</span>
                                    <span class="badge-sim">{sim_pct}% match</span>
                                </div>
                            </div>
                            <div style="margin-bottom: 8px;">
                                {genre_pills}
                            </div>
                            <div class="why-box">
                                <strong>💡 Why Recommended:</strong> {item['why_recommended']}
                            </div>
                            <p style="margin: 6px 0; font-size: 0.88rem; color: #9ca3af; line-height: 1.4;">
                                {item['overview']}
                            </p>
                            <div style="font-size: 0.8rem; color: #64748b; margin-top: 6px;">
                                <strong>Director:</strong> {item['director']} &nbsp;|&nbsp; 
                                <strong>Language:</strong> {item['language']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    except Exception as e:
        st.error(f"Error computing recommendations: {str(e)}")

with tab_analytics:
    st.markdown("### 🔬 Interpretability & TF-IDF Vector Space Analysis")
    st.markdown(
        """
        Explainability is crucial in recommender systems. By projecting text into a TF-IDF vector space,
        we can compute the exact element-wise vector products to inspect the words driving high cosine similarity.
        """
    )

    if not recs_df.empty:
        chart_data = recs_df[["title", "similarity_score", "hybrid_score"]].set_index("title")
        st.subheader("Cosine Similarity vs. Final Hybrid Ranking Score")
        st.bar_chart(chart_data)

        top_1 = recs_df.iloc[0]
        st.subheader(f"Vector Overlap Breakdown: '{selected_title}' ↔ '{top_1['title']}'")
        
        seed_idx = recommender.title_to_idx[selected_title.lower()]
        target_idx = recommender.title_to_idx[top_1['title'].lower()]
        
        row1 = recommender.tfidf_matrix[seed_idx].toarray().flatten()
        row2 = recommender.tfidf_matrix[target_idx].toarray().flatten()
        overlap = row1 * row2
        top_indices = np.argsort(overlap)[::-1][:10]

        term_breakdown = []
        for i in top_indices:
            val = overlap[i]
            if val > 0.0001:
                term_breakdown.append({
                    "Feature / Keyword": recommender.feature_names[i],
                    "Overlap Product (Contribution)": round(float(val), 5),
                    "Seed TF-IDF": round(float(row1[i]), 4),
                    "Target TF-IDF": round(float(row2[i]), 4),
                })
        
        if term_breakdown:
            st.dataframe(pd.DataFrame(term_breakdown), use_container_width=True)
        else:
            st.info("No single dominant keyword; recommendation driven by combined multi-field tags.")

with tab_eval:
    st.markdown("### 🎯 Offline Benchmark Evaluation (Section 10 of Guide)")
    st.markdown(
        """
        To validate recommendation quality without live user interactions, we measured 
        <strong>Precision@5</strong>, <strong>Precision@10</strong>, <strong>Intra-List Diversity</strong>, 
        and <strong>Catalog Coverage</strong> across 20 curated benchmark titles.
        """,
        unsafe_allow_html=True,
    )

    if eval_report:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision @ 5", f"{eval_report['mean_precision@5'] * 100:.1f}%")
        m2.metric("Precision @ 10", f"{eval_report['mean_precision@10'] * 100:.1f}%")
        m3.metric("Genre Alignment", f"{eval_report['mean_genre_alignment@5'] * 100:.1f}%")
        m4.metric("Avg Latency", f"{eval_report['mean_query_latency_ms']:.1f} ms")

        m5, m6, m7 = st.columns(3)
        m5.metric("Intra-List Diversity", f"{eval_report['intra_list_diversity@5']:.3f}", help="Scale: 0 (monolithic) to 1.0 (highly diverse)")
        m6.metric("Catalog Coverage", f"{eval_report['catalog_coverage_pct']:.1f}%", help="% of catalog reachable via top-10 lists")
        m7.metric("Catalog Size", f"{eval_report['catalog_size']} movies")

        st.markdown("---")
        st.subheader("Per-Seed Benchmark Performance")
        bench_df = pd.DataFrame(eval_report["detailed_seeds"])
        st.dataframe(bench_df, use_container_width=True)

        st.subheader("Documented Failure Modes & Trade-Offs")
        st.markdown(
            """
            1. **Cross-Lingual Vocabulary Disconnect:** Non-English films with short translated overviews occasionally miss thematic links to Hollywood titles unless bridged by deliberate genre/keyword tags.
            2. **Sequel / Franchise Dominance:** Unconstrained TF-IDF tends to fill all top slots with sequels (e.g., *The Matrix* recommending *The Matrix Reloaded*). The optional **Intra-List Diversity** filter resolves this.
            3. **Serendipity vs. Thematic Precision:** Highly specific TF-IDF vectors excel at strict similarity but can lack accidental discoverability, which can be mitigated with hybrid popularity weighting.
            """
        )
    else:
        st.info("Run `python evaluate.py` to generate the latest evaluation report.")

with tab_interview:
    st.markdown("### 📚 Architecture Walkthrough & Technical Interview Reference")

    st.markdown(
        r"""
        #### 1. Architecture Flow
        ```text
        Movie Dataset (112 titles) 
          ↳ Cleaning (impute nulls, lowercase, remove punctuation)
          ↳ Feature Engineering (weighted tags: genres x3, keywords x2, director x2)
          ↳ NLP / TF-IDF Vectorizer (stop_words='english', n-grams=(1,2), max_features=20,000)
          ↳ Cosine Similarity Matrix (112 x 112 pairwise angle computation)
          ↳ Multi-Constraint Filter (genre, language, year range, minimum rating)
          ↳ Hybrid Ranking (α · CosineSimilarity + β · NormalizedRating)
          ↳ Top-N Recommendations + Explainability Badges (Streamlit UI)
        ```
        ---
        #### 2. Section 13: Core Technical Interview Questions & Answers
        
        **Q1: Why content-based recommendation?**  
        *Answer:* Content-based filtering does not require user interaction history or ratings matrices. It operates solely on item metadata, making it completely immune to the **user cold-start problem** (recommending to brand new users immediately) and capable of recommending niche, newly released, or long-tail movies with zero prior views.

        **Q2: Why TF-IDF over simple Bag-of-Words (CountVectorizer)?**  
        *Answer:* Bag-of-Words weights words purely by frequency, meaning generic high-frequency words (e.g. *film*, *story*, *man*) would skew similarity scores. TF-IDF applies Inverse Document Frequency ($IDF = \log(N / DF)$) to penalize terms that appear everywhere while heavily rewarding rare, informative terms (e.g. *tesseract*, *subconscious*, *replicant*).

        **Q3: What does Cosine Similarity measure?**  
        *Answer:* Cosine similarity measures the cosine of the angle between two multi-dimensional feature vectors:
        $$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$
        It evaluates the orientation and thematic alignment of two documents regardless of document length. A short, focused synopsis can have high cosine similarity with a longer synopsis if their relative term proportions match.

        **Q4: How would you solve the Cold Start problem?**  
        *Answer:*  
        - *User Cold-Start:* Present an onboarding survey (e.g. "Select 3 movies you love") to immediately establish a content profile vector without needing watch history.
        - *Item Cold-Start:* Content-based filtering inherently solves this. As soon as metadata (synopsis, director, tags) is available, a new movie can be vectorized and recommended immediately.

        **Q5: How would you evaluate recommendations offline vs. online?**  
        *Answer:*  
        - *Offline:* Precision@K, Recall@K, Mean Reciprocal Rank (MRR), Intra-List Diversity (ILD), and Catalog Coverage on a curated benchmark set.  
        - *Online:* A/B testing measuring Click-Through Rate (CTR) on recommended carousels, Mean Watch Time / Completion Rate, and 30-day Subscriber Retention.

        **Q6: How would you add Collaborative Filtering?**  
        *Answer:* By collecting implicit feedback (clicks, watch duration) and explicit feedback (user ratings). We can compute item-item or user-user collaborative filtering using Matrix Factorization (Truncated SVD / ALS) or Neural Collaborative Filtering, blending CF scores with content scores in a **Hybrid Recommender** pipeline.

        **Q7: How would you personalize recommendations for individual users?**  
        *Answer:* Aggregate the TF-IDF vectors of all movies a user has watched and rated positively into a composite **User Profile Vector** (weighted by user rating). Then compute the cosine similarity between the User Profile Vector and unconsumed catalog items.
        
        ---
        #### 3. Section 12: Resume Bullet Points
        - *Built a content-based movie recommendation system using Python, Pandas, NLP and TF-IDF to generate Top-N recommendations from movie metadata.*
        - *Applied cosine similarity and added genre/language/year filtering for more relevant recommendations.*
        - *Developed an optional Streamlit interface for interactive movie discovery.*
        """
    )
