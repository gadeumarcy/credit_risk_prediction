from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# Chargement du modèle et des ressources
modele = joblib.load("modele_credit.pkl")
scaler = joblib.load("scaler.pkl")
colonnes = joblib.load("colonnes.pkl")

# Définition du modèle d’entrée
class DonneesClient(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

# Création de l’app FastAPI
app = FastAPI()

@app.post("/predire")
def predire(donnees: DonneesClient):
    try:
        # Convertir les données en dictionnaire puis en DataFrame
        data_dict = donnees.model_dump()
        df = pd.DataFrame([data_dict])

        # Encodage
        df = pd.get_dummies(df)
        for col in colonnes:
            if col not in df.columns:
                df[col] = 0
        df = df[colonnes]

        # Standardisation
        df_scaled = scaler.transform(df)

        # Prédiction
        prediction = int(modele.predict(df_scaled)[0])
        proba = float(modele.predict_proba(df_scaled)[0][1])

        resultat = "⚠️ Ce client est à RISQUE" if prediction == 1 else "✅ Ce client est FIABLE"

        return {
            "résultat": resultat,
            "probabilité_de_risque": round(proba, 2)
        }

    except Exception as e:
        return {"erreur": str(e)}
