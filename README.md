# 🎬 CineMatch AI: Explainable Movie Recommendation & Personalization Engine
**An End-to-End Machine Learning & NLP Portfolio Project**  
*Built by [Ranjith Kumar](https://github.com/Ranjith233) • Python, Pandas, Scikit-Learn, TF-IDF, Streamlit*

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%26%20Cosine%20Sim-blue?style=flat)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Precision@5](https://img.shields.io/badge/Precision%405-98.0%25-success?style=flat)
![Status](https://img.shields.io/badge/Status-Validated%20%26%20Tested-10B981?style=flat)

---

## 📌 1. Why I Built This Project

Streaming services host catalogs with tens of thousands of movies. While collaborative filtering (user-user or matrix factorization) powers many platforms, it completely breaks down in two critical industry scenarios:
1. **The Item Cold-Start Problem:** When an OTT platform licenses or releases a brand-new movie, zero user ratings or watch logs exist. Collaborative filtering cannot recommend it.
2. **The Black-Box Problem:** Modern deep recommenders rarely explain *why* an item was surfaced, leading to low user trust and high bounce rates on home carousels.

I engineered this project to demonstrate how an **explainable content-based recommendation system** can be designed, optimized, and evaluated from scratch using natural language processing (NLP), multi-field feature weighting, and cosine similarity. 

The system enables **instant, cold-start recommendations** for new movies, provides **transparent human-readable explanations** for every recommendation, and incorporates **business filters** (genre, language, release year, IMDb rating) alongside **hybrid ranking signals**.

---

## 🚀 Key Results & Highlights

- **98.0% Precision@5** and **96.0% Precision@10** achieved across a 20-seed benchmark evaluation suite.
- **97.0% Genre Alignment** in top recommendations without sacrificing theme diversity.
- **36.1 ms Average Query Latency**, fast enough for real-time autocompletion and interactive web discovery.
- **Explainability Engine:** Dissects overlapping TF-IDF vectors into clear user badges (e.g. *"Directed by Christopher Nolan • Shared genres: Sci-Fi, Action • Thematic keywords: espionage, inversion"*).
- **Graceful Typo & Fuzzy Matching:** Automatically resolves imperfect queries (e.g. `"dark night"` $\to$ *The Dark Knight*, `"intersteler"` $\to$ *Interstellar*).
- **Intra-List Diversity Control:** Down-weights redundant franchise sequels to avoid recommendation echo chambers.

---

## 📂 Project Architecture & File Hierarchy

```text
movie-recommender/
├── data/
│   ├── movies.csv                   # Clean catalog with rich metadata (112 curated films)
│   ├── evaluation_benchmark.csv     # Benchmark seeds with ground-truth target attributes
│   ├── evaluation_report.json       # Persisted benchmark evaluation metrics & audit trail
│   └── build_dataset.py             # Script to regenerate and expand movie datasets
├── notebooks/
│   └── 01_recommender.ipynb         # End-to-end exploratory analysis, modeling & evaluation
├── src/
│   ├── __init__.py                  # Package exports
│   ├── preprocess.py                # Text normalization, entity concatenation & weighted tags
│   └── recommender.py               # Recommender class, TF-IDF, Cosine Sim, explainability
├── app.py                           # Interactive Streamlit discovery dashboard
├── evaluate.py                      # Automated offline evaluation suite (Precision@K, Latency)
├── test_recommender.py              # Unit & integration sanity verification tests
├── requirements.txt                 # Project dependencies
└── README.md                        # Portfolio documentation & interview guide
```

### End-to-End Pipeline Workflow (PDF Section 2)

```text
┌────────────────┐     ┌────────────────┐     ┌───────────────────────┐
│  Movie Dataset │ ──> │ Cleaning & Pre-│ ──> │ Multi-Field Weighted  │
│  (112 titles)  │     │ processing     │     │ Feature Engineering   │
└────────────────┘     └────────────────┘     └───────────────────────┘
                                                          │
                                                          ▼
┌────────────────┐     ┌────────────────┐     ┌───────────────────────┐
│  Streamlit UI  │ <── │ Top-N Hybrid   │ <── │ TF-IDF Vector Space   │
│  Interactive   │     │ Ranking & Ex-  │     │ & Cosine Similarity   │
│  Discovery App │     │ plainability   │     │ Matrix (NxN)          │
└────────────────┘     └────────────────┘     └───────────────────────┘
```

---

## 🛠️ Data Preprocessing & Strategic Feature Weighting

### The Limitation of Baseline Concatenation
A naive content-based baseline (PDF Section 4) combines fields equally:
$$\text{tags} = \text{genres} + \text{keywords} + \text{overview} + \text{director}$$

In practice, a plot overview contains 60+ generic narrative tokens (*"man"*, *"police"*, *"journey"*, *"life"*), while critical categorical anchors (*"Sci-Fi"*, *"Christopher Nolan"*) only appear once. This causes vectorizers to overfit to incidental plot trivia rather than core cinematic identity.

### Our Solution: Multi-Field Term Weighting (PDF Section 8)
In `src/preprocess.py`, we implement term multiplication to scale the term frequencies (TF) of deliberate metadata:
- **Genre Weight ($\times 3$):** Strongly anchors narrative genre consistency.
- **Keyword Weight ($\times 2$):** Emphasizes thematic tags (*"dream"*, *"cyberpunk"*, *"mafia"*).
- **Director Weight ($\times 2$):** Captures distinct auteur styles and cinematic voices.
- **Entity Concatenation:** Converts multi-word names into single tokens (`"Christopher Nolan"` $\to$ `"ChristopherNolan"`) to prevent false positive correlations with unrelated people sharing first or last names.

---

## 🧮 Mathematical Formulation

### 1. TF-IDF Vectorization
For a term $t$ in a document $d$ within catalog corpus $D$:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

Where:
- $\text{TF}(t, d)$ is the frequency of term $t$ in movie document $d$.
- $\text{IDF}(t, D)$ suppresses ubiquitous tokens while boosting discriminative, rare keywords.

### 2. Cosine Similarity
To measure the orientation of two movie vectors $\vec{u}$ and $\vec{v}$ independent of length:
$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2} = \frac{\sum_{i=1}^{m} u_i v_i}{\sqrt{\sum_{i=1}^m u_i^2} \sqrt{\sum_{i=1}^m v_i^2}}$$

### 3. Hybrid Ranking Score
To balance thematic similarity with catalog popularity and critical quality:
$$\text{Score}_{\text{final}} = (1 - w_{\text{pop}} - w_{\text{rat}}) \cdot \text{Sim}(\vec{u}, \vec{v}) + w_{\text{pop}} \cdot \frac{\text{Popularity}}{100} + w_{\text{rat}} \cdot \frac{\text{Rating}}{10}$$

---

## 🎯 Model Evaluation Benchmark (PDF Section 10)

We evaluated the recommender against a 20-seed benchmark catalog representing diverse genres, eras, and foreign-language cinema.

| Evaluation Metric | Value | Benchmark Description |
| :--- | :---: | :--- |
| **Precision @ 5** | **98.0%** | Share of top-5 recommendations matching target genre/themes |
| **Precision @ 10** | **96.0%** | Share of top-10 recommendations matching target genre/themes |
| **Genre Alignment @ 5** | **97.0%** | Direct genre overlap between seed and top recommendations |
| **Intra-List Diversity (ILD)** | **0.916** | Average pairwise cosine distance (Scale: 0 = clones, 1 = max diversity) |
| **Catalog Coverage** | **84.8%** | Percentage of entire catalog discoverable across benchmark queries |
| **Mean Query Latency** | **36.09 ms** | End-to-end execution time per recommendation query |

### Sample Benchmark Results

| Seed Movie | Precision@5 | Precision@10 | Top #1 Recommendation | Cosine Sim |
| :--- | :---: | :---: | :--- | :---: |
| **Inception** | 100% | 100% | *Tenet* | 0.132 |
| **The Dark Knight** | 100% | 100% | *Batman Begins* | 0.242 |
| **The Godfather** | 100% | 100% | *The Godfather: Part II* | 0.350 |
| **Pulp Fiction** | 100% | 100% | *Django Unchained* | 0.094 |
| **Spirited Away** | 100% | 100% | *Howl's Moving Castle* | 0.269 |
| **Parasite** | 100% | 100% | *Memories of Murder* | 0.305 |
| **The Matrix** | 100% | 90% | *The Matrix Reloaded* | 0.518 |
| **Whiplash** | 80% | 50% | *La La Land* | 0.225 |

### Documented Failure Modes & Trade-Offs (PDF Section 10)
1. **Cross-Lingual Vocabulary Asymmetry:** Non-English masterworks (*e.g., Spirited Away, City of God*) with brief translated synopses risk missing thematic links to English counterparts unless deliberate English thematic keywords are tagged.
2. **Franchise Monopolization:** Pure TF-IDF tends to fill all top recommendation slots with sequels (*e.g., The Matrix Reloaded, The Matrix Revolutions*). We introduced an **Intra-List Diversity filter** that caps any single creator/franchise at 2 items in top-5 recommendations.
3. **The Serendipity Trade-off:** High feature weighting guarantees relevance but can occasionally reduce unexpected discoveries ("happy accidents"). The hybrid rating/popularity slider re-injects serendipity.

---

## 💻 Quickstart: How to Run

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
```

### 2. Run the Streamlit Discovery App (PDF Section 9)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test interactive movie recommendations, custom filters, and vector space breakdowns.

### 3. Run the Automated Evaluation Benchmark (PDF Section 10)
```bash
python evaluate.py
```

### 4. Explore the Interactive Jupyter Notebook
```bash
jupyter lab notebooks/01_recommender.ipynb
```

---

## 📝 Section 12: Resume Bullet Points

If you are showcasing this project on your resume, use these recruiter-tested bullet points:

- **Built an explainable content-based movie recommendation system** using Python, Pandas, NLP, and Scikit-Learn TF-IDF, achieving **98.0% Precision@5** across a 20-seed benchmark catalog.
- **Engineered multi-field weighted text representations** and cosine similarity matrices, integrating genre, language, and release year filtering to resolve item cold-start discovery.
- **Developed an interactive Streamlit web dashboard** featuring real-time metadata filtering, hybrid quality-score ranking, and vector-level explainability for recommendation transparency.

---

## 💡 Section 13: Technical Interview Preparation (Q&A)

Here are comprehensive answers to the 7 core interview questions outlined in Section 13 of the project guide:

### Q1: Why content-based recommendation?
**Answer:** Content-based recommendation computes item similarity using inherent item metadata (synopsis, genre, director, actors) rather than past user interaction matrices. This provides three major strategic advantages:
1. **Zero User Cold-Start:** Recommends items to brand-new users immediately after they select even one preferred movie.
2. **Zero Item Cold-Start:** Newly licensed or niche titles can be recommended the moment their metadata is uploaded.
3. **Transparent Explainability:** Recommendations are completely interpretable (e.g. *"Recommended because of shared Sci-Fi genre and director Christopher Nolan"*).

### Q2: Why TF-IDF over CountVectorizer (Bag-of-Words)?
**Answer:** `CountVectorizer` counts raw term occurrences. High-frequency words that appear across almost all movie overviews (e.g., *"man"*, *"young"*, *"story"*, *"world"*) dominate the vector magnitude, introducing noise. TF-IDF discounts frequent catalog words via **Inverse Document Frequency ($IDF = \log(N / DF)$)** while boosting rare, highly discriminative descriptors (e.g., *"subconscious"*, *"replicant"*, *"tesseract"*, *"mafia"*).

### Q3: What does Cosine Similarity measure?
**Answer:** Cosine similarity measures the cosine of the angle between two vectors in a high-dimensional space:
$$\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$
Because it normalizes by Euclidean vector magnitude, it evaluates **thematic and proportional similarity regardless of document length**. A 2-sentence synopsis and a 500-word plot summary will have high cosine similarity if they share the same proportion of distinctive keywords.

### Q4: How would you solve the Cold Start problem?
**Answer:**
- **Item Cold-Start:** Content-based filtering is the industry standard solution. A newly ingested movie needs zero watch history; its TF-IDF vector is computed from metadata, allowing it to be surfaced immediately.
- **User Cold-Start:** Present an onboarding survey during signup (e.g., "Select 3 movies or genres you enjoy"). Synthesize these selections into an initial user profile vector and recommend top cosine matches.

### Q5: How would you evaluate recommendations offline vs. online?
**Answer:**
- **Offline Evaluation:** Curate an evaluation benchmark set of seed movies with known thematic ground truths. Measure **Precision@K**, **Recall@K**, **Intra-List Diversity (ILD)**, and **Catalog Coverage**.
- **Online Evaluation (A/B Testing):** Split live users into control and test buckets. Measure business KPIs: **Click-Through Rate (CTR)** on recommendation carousels, **Average Watch Time / Completion Rate**, and **30-day Subscriber Retention**.

### Q6: How would you add Collaborative Filtering?
**Answer:** By capturing implicit user feedback (clicks, watch time, completion rate) and explicit user feedback (thumbs up, star ratings). We construct a sparse User-Item interaction matrix and apply:
1. **Matrix Factorization (Truncated SVD / Alternating Least Squares - ALS)** to decompose users and items into latent embedding vectors.
2. **Hybrid Blending:** Compute a weighted sum of content-based cosine similarity and collaborative filtering prediction:
$$\text{Final Score} = \alpha \cdot \text{Score}_{\text{content}} + (1 - \alpha) \cdot \text{Score}_{\text{CF}}$$

### Q7: How would you personalize recommendations for individual users?
**Answer:** Create a dynamic **User Profile Vector** ($\vec{U}$) representing the weighted average of the TF-IDF vectors of all movies the user has watched, weighted by their explicit ratings or watch completion percentages:
$$\vec{U} = \frac{\sum_{i \in \text{Watched}} w_i \cdot \vec{V}_i}{\sum_{i \in \text{Watched}} w_i}$$
Then calculate the cosine similarity between $\vec{U}$ and all unconsumed catalog movies to populate a customized "Recommended For You" feed.

---

## 📜 License & Acknowledgments
- Developed as part of the **OTT / Entertainment Analytics Portfolio Series**.
- Movie metadata curated for educational and portfolio demonstration purposes.
- Free for use and adaptation under the [MIT License](LICENSE).
