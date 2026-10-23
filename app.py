from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="FlipItNews | News Classifier",
    page_icon="📰",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "best_traditional_model.joblib")
    vectorizer = joblib.load(MODEL_DIR / "best_vectorizer.joblib")
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    return model, vectorizer, encoder

model, vectorizer, encoder = load_artifacts()

st.title("📰 FlipItNews")
st.subheader("Intelligent News Category Classification")
st.write(
    "Enter a news article to classify it into one of five categories: "
    "business, entertainment, politics, sport, or tech."
)

with st.sidebar:
    st.header("About the project")
    st.write("**Algorithm:** Multinomial Naive Bayes")
    st.write("**Features:** Bag of Words (BoW)")
    st.write("**Categories:** 5")

    results_file = OUTPUT_DIR / "model_comparison.csv"
    if results_file.exists():
        results = pd.read_csv(results_file)
        st.write("**Validation model comparison**")
        st.dataframe(results, use_container_width=True, hide_index=True)

article = st.text_area(
    "Enter the news article",
    height=220,
    placeholder="Paste the news headline or article text here..."
)

if st.button("Classify News", type="primary", use_container_width=True):
    if not article.strip():
        st.warning("Please enter some news text first.")
    else:
        features = vectorizer.transform([article])
        prediction = model.predict(features)[0]
        category = encoder.inverse_transform([prediction])[0]

        st.success(f"Predicted category: {category.title()}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            probability_df = pd.DataFrame({
                "Category": encoder.classes_,
                "Probability": probabilities
            }).sort_values("Probability", ascending=False)

            st.write("### Category confidence")
            st.bar_chart(
                probability_df.set_index("Category")["Probability"]
            )
            st.caption(
                "These are model probabilities, not guaranteed measures "
                "of correctness."
            )

st.divider()
st.caption("FlipItNews | NLP Text Classification Project")
