import streamlit as st
import pandas as pd
import joblib
from PIL import Image



# Chargement du modèle et des colonnes
modele = joblib.load("modele_credit.pkl")
scaler = joblib.load("scaler.pkl")
colonnes = joblib.load("colonnes.pkl")

st.set_page_config(page_title="Prédiction du Risque de Crédit", layout="centered")

st.title("💳 Prédiction du Risque de Crédit")

st.markdown("<h1 style='text-align: center;'>Prédiction du Risque de Crédit</h1>", unsafe_allow_html=True)

image = Image.open("logo.png")
st.image(image, width=200, use_column_width=False)
# Formulaire utilisateur
with st.form("formulaire_credit"):
    st.subheader("Informations Client")

    age = st.number_input("Âge du client", min_value=18, max_value=100)
    revenu = st.number_input("Revenu annuel", step=1000)
    home = st.selectbox("Type de logement", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    emp_length = st.number_input("Années d'expérience", min_value=0.0, max_value=100.0)
    intent = st.selectbox("Motif du prêt", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    grade = st.selectbox("Note du prêt", ["A", "B", "C", "D", "E", "F", "G"])
    montant = st.number_input("Montant du prêt")
    interet = st.number_input("Taux d'intérêt (%)", step=0.01)
    percent_income = st.number_input("% du prêt par rapport au revenu", step=0.01)
    default = st.selectbox("Défaut dans le passé ?", ["Y", "N"])
    cred_length = st.number_input("Longueur historique du crédit (en années)", min_value=0)

    submitted = st.form_submit_button("Prédire le risque")

if submitted:
    try:
        donnees = {
            'person_age': age,
            'person_income': revenu,
            'person_home_ownership': home,
            'person_emp_length': emp_length,
            'loan_intent': intent,
            'loan_grade': grade,
            'loan_amnt': montant,
            'loan_int_rate': interet,
            'loan_percent_income': percent_income,
            'cb_person_default_on_file': default,
            'cb_person_cred_hist_length': cred_length
        }

        df = pd.DataFrame([donnees])
        df = pd.get_dummies(df)

        # Ajouter colonnes manquantes
        for col in colonnes:
            if col not in df.columns:
                df[col] = 0

        df = df[colonnes]

        # Standardiser
        donnees_scaled = scaler.transform(df)

        # Prédiction
        prediction = int(modele.predict(donnees_scaled)[0])
        proba = float(modele.predict_proba(donnees_scaled)[0][1])

        if prediction == 1:
            st.error(f"⚠️ Ce client est à RISQUE — Probabilité de défaut : {round(proba * 100, 2)} %")
        else:
            st.success(f"✅ Ce client est FIABLE — Probabilité de défaut : {round(proba * 100, 2)} %")

    except Exception as e:
        st.warning(f"Erreur : {str(e)}")
