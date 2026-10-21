import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import joblib
import numpy as np
from src.preprocess import clean_text


# -----------------------------------------
# LOAD MODELS 
# -----------------------------------------
models = {
    "Logistic Regression": joblib.load("models/logreg_model.joblib"),
    "Random Forest": joblib.load("models/rf_model.joblib"),
    "Naive Bayes": joblib.load("models/nb_model.joblib"),
    "SVM": joblib.load("models/svm_model.joblib")
}

# -----------------------------------------
# UI
# -----------------------------------------
st.title("📰 Fake News Detection")

model_choice = st.selectbox(
    "Select Model:",
    ["Compare All Models"] + list(models.keys())
)

text_input = st.text_area("✍️ Enter news text:")

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text_input)

        # -------------------------------------
        # SINGLE MODEL MODE
        # -------------------------------------
        if model_choice != "Compare All Models":
            model = models[model_choice]

            pred = model.predict([cleaned])[0]
            label = "Fake" if pred == 1 else "Real"

            try:
                prob_fake = model.predict_proba([cleaned])[0][1]
            except:
                prob_fake = np.nan

            st.subheader(f"🔍 Result using {model_choice}")
            st.write(f"**Prediction:** {label}")
            st.write(f"**Probability Fake:** {prob_fake:.2f}")

        # -------------------------------------
        # COMPARE ALL MODELS (NO pyarrow)
        # -------------------------------------
        else:
            st.subheader("📊 Comparing All Models")

            predictions = []
            fake_probs = []

            for name, model in models.items():
                pred = model.predict([cleaned])[0]
                label = "Fake" if pred == 1 else "Real"

                try:
                    prob = model.predict_proba([cleaned])[0][1]
                except:
                    prob = np.nan

                predictions.append((name, label, prob))
                fake_probs.append(prob)

            # Print results manually (NO dataframe)
            for name, label, prob in predictions:
                st.write(f"### {name}")
                st.write(f"- Prediction: **{label}**")
                st.write(f"- Fake Probability: **{prob:.2f}**")
                st.write("---")

            # Majority vote
            votes = [p[1] for p in predictions]
            final = max(set(votes), key=votes.count)

            st.subheader("🧠 Final Decision (Majority Vote)")
            st.success(f"**{final}**")
