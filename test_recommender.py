from src.preprocess import load_and_preprocess_catalog
from src.recommender import ContentBasedRecommender

df = load_and_preprocess_catalog("data/movies.csv", weighted=True)
rec = ContentBasedRecommender(df)

print("=== Test 1: Seed 'Inception' ===")
recs = rec.recommend("Inception", n=5)
for i, r in recs.iterrows():
    print(f"{i+1}. {r['title']} ({r['release_year']}) [Sim: {r['similarity_score']:.3f}] - {r['genres']}")
    print(f"   Why: {r['why_recommended']}")

print("\n=== Test 2: Seed 'The Godfather' with Genre Filter 'Crime' ===")
recs = rec.recommend("The Godfather", n=4, genre_filter="Crime")
for i, r in recs.iterrows():
    print(f"{i+1}. {r['title']} ({r['release_year']}) [Sim: {r['similarity_score']:.3f}] - {r['director']}")
    print(f"   Why: {r['why_recommended']}")

print("\n=== Test 3: Fuzzy Matching for Typos ===")
t1, _, _ = rec.find_title("dark night")
print(f"Query 'dark night' -> Found: '{t1}'")

t2, _, _ = rec.find_title("intersteler")
print(f"Query 'intersteler' -> Found: '{t2}'")

t3, _, _ = rec.find_title("spirit away")
print(f"Query 'spirit away' -> Found: '{t3}'")
