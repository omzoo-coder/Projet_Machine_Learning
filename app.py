import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Projet Machine Learning", layout="centered")

# dossier où se trouvent les fichiers joblib
DOSSIER_MODELES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# =====================================================================
# Fonctions communes aux 4 modèles
# =====================================================================
@st.cache_resource
def charger(prefix):
    """Charge le modèle, le scaler, les encodeurs et les infos (meta) d'un modèle."""
    def chemin(nom):
        return os.path.join(DOSSIER_MODELES, f"{prefix}_{nom}.joblib")

    model = joblib.load(chemin("model"))
    scaler = joblib.load(chemin("scaler"))
    encoders = joblib.load(chemin("encoders"))
    meta = joblib.load(chemin("meta"))
    return model, scaler, encoders, meta


def noms_classes(encoders, meta, cible):
    """Noms des classes (classification) : on les lit dans l'encodeur de la cible."""
    if cible and isinstance(encoders, dict) and cible in encoders:
        return [str(c) for c in encoders[cible].classes_]
    return list(meta.get("classnames", []))


def preparer(df, encoders, scaler, meta):
    """Encode les variables catégorielles puis normalise, comme à l'entraînement."""
    X = df[meta["feature_cols"]].copy()

    if X.isna().any().any():
        raise ValueError("Le fichier contient des valeurs manquantes dans les colonnes utilisées.")

    for col in meta["cat_features"]:
        X[col] = X[col].astype(str).str.strip()
        inconnues = sorted(set(X[col]) - set(str(v) for v in meta["cat_options"][col]))
        if inconnues:
            raise ValueError(f"Valeurs inconnues dans la colonne '{col}' : {inconnues[:5]}")
        # encodeurs : dictionnaire (par colonne) ou liste (dans l'ordre des colonnes catégorielles)
        if isinstance(encoders, dict):
            encodeur = encoders[col]
        else:
            encodeur = encoders[meta["cat_features"].index(col)]
        X[col] = encodeur.transform(X[col])

    return scaler.transform(X.values.astype(float))


def predire(df, prefix, cible=None):
    """Retourne (prédictions, probabilités). Les probabilités existent seulement en classification."""
    model, scaler, encoders, meta = charger(prefix)
    X = preparer(df, encoders, scaler, meta)
    pred = model.predict(X)

    if cible is None:  # régression
        return pd.Series(pred, index=df.index).round(2), None

    # classification
    noms = noms_classes(encoders, meta, cible)
    labels = pd.Series([noms[int(p)] for p in pred], index=df.index)
    probas = pd.DataFrame(model.predict_proba(X) * 100,
                          columns=[noms[int(c)] for c in model.classes_],
                          index=df.index)
    return labels, probas


def champ_numerique(col, meta, prefix):
    """Champ de saisie pour une variable numérique (bornes = valeurs vues à l'entraînement)."""
    nom = col.replace("_", " ")
    cle = f"s_{prefix}_{col}"
    bornes = meta.get("num_ranges", {}).get(col)
    if bornes:
        mini, maxi = float(bornes[0]), float(bornes[1])
        if mini.is_integer() and maxi.is_integer():
            return st.number_input(nom, min_value=int(mini), max_value=int(maxi),
                                   value=int((mini + maxi) // 2), step=1, key=cle)
        return st.number_input(nom, min_value=mini, max_value=maxi,
                               value=round((mini + maxi) / 2, 2), step=0.01, key=cle)
    return st.number_input(nom, value=0.0, key=cle)


def exemple_csv(meta):
    """Une ligne d'exemple pour le fichier CSV modèle."""
    ligne = {}
    for col in meta["feature_cols"]:
        if col in meta["cat_features"]:
            ligne[col] = meta["cat_options"][col][0]
        else:
            bornes = meta.get("num_ranges", {}).get(col)
            ligne[col] = round((bornes[0] + bornes[1]) / 2, 2) if bornes else 0
    return pd.DataFrame([ligne])


def afficher_modele(titre, prefix, cible=None, unite="", description=""):
    """Page d'un modèle : onglet Prédiction simple + onglet Prédiction multiple."""
    st.title(titre)
    if description:
        st.write(description)

    model, scaler, encoders, meta = charger(prefix)
    onglet_simple, onglet_multiple = st.tabs(["Prédiction simple", "Prédiction multiple"])

    # ---------- Prédiction simple ----------
    with onglet_simple:
        with st.form(f"formulaire_{prefix}"):
            valeurs = {}
            colonnes = st.columns(2)
            for i, col in enumerate(meta["feature_cols"]):
                with colonnes[i % 2]:
                    if col in meta["cat_features"]:
                        valeurs[col] = st.selectbox(col.replace("_", " "),
                                                    meta["cat_options"][col],
                                                    key=f"s_{prefix}_{col}")
                    else:
                        valeurs[col] = champ_numerique(col, meta, prefix)
            envoye = st.form_submit_button("Prédire")

        if envoye:
            try:
                df = pd.DataFrame([valeurs])
                resultat, probas = predire(df, prefix, cible)
                if cible is None:
                    st.metric(f"{meta['target'].replace('_', ' ')} prédit",
                              f"{resultat.iloc[0]:,.2f}{unite}")
                else:
                    st.success(f"Prédiction : {resultat.iloc[0]}")
                    st.caption("Probabilité de chaque classe (%)")
                    st.bar_chart(probas.iloc[0].round(1))
            except Exception as e:
                st.error(f"Erreur : {e}")

    # ---------- Prédiction multiple ----------
    with onglet_multiple:
        st.write("Le fichier CSV doit contenir ces colonnes :")
        st.code(", ".join(meta["feature_cols"]))
        st.download_button("Télécharger un fichier CSV modèle",
                           exemple_csv(meta).to_csv(index=False).encode("utf-8"),
                           file_name=f"modele_{prefix}.csv", mime="text/csv",
                           key=f"modele_{prefix}")
        fichier = st.file_uploader("Choisis ton fichier CSV", type="csv", key=f"upload_{prefix}")

        if fichier is not None:
            try:
                df = pd.read_csv(fichier)
                manquantes = [c for c in meta["feature_cols"] if c not in df.columns]
                if manquantes:
                    st.error(f"Colonnes manquantes dans le fichier : {manquantes}")
                else:
                    resultat, probas = predire(df, prefix, cible)
                    sortie = df.copy()
                    if cible is None:
                        sortie[f"{meta['target']}_prediction"] = resultat
                    else:
                        sortie["Prediction"] = resultat
                        sortie["Probabilite (%)"] = probas.max(axis=1).round(1)
                    st.dataframe(sortie)
                    st.download_button("Télécharger les prédictions (CSV)",
                                       sortie.to_csv(index=False).encode("utf-8"),
                                       file_name=f"predictions_{prefix}.csv", mime="text/csv",
                                       key=f"dl_{prefix}")
            except Exception as e:
                st.error(f"Erreur : {e}")


# =====================================================================
# Menu et pages
# =====================================================================
choix = st.sidebar.radio("Choisis un modèle",
                         ["Accueil", "Telephone", "Car Data", "Taxi Trip", "Car Dataset"])

if choix == "Accueil":
    st.title("Projet Machine Learning")
    st.write("Cette application regroupe 4 modèles de machine learning. "
             "Choisis un modèle dans le menu à gauche.")
    st.markdown("""
| Modèle | Type | Variable cible |
|---|---|---|
| Telephone | Classification | `etat` |
| Car Data | Classification | `Venant_Occasion` |
| Taxi Trip | Régression | `Trip_Price` |
| Car Dataset | Régression | `Selling_Price` |
""")
    st.info("Chaque modèle a deux interfaces : **Prédiction simple** (une seule observation) "
            "et **Prédiction multiple** (un fichier CSV).")

elif choix == "Telephone":
    afficher_modele("Telephone - Classification", "telephone", cible="etat",
                    description="Prédit l'état du téléphone (variable `etat`).")

elif choix == "Car Data":
    afficher_modele("Car Data - Classification", "car_data", cible="Venant_Occasion",
                    description="Prédit si la voiture est **Venant** ou **D'occasion**.")

elif choix == "Taxi Trip":
    afficher_modele("Taxi Trip - Régression", "taxi_trip", unite=" $",
                    description="Prédit le prix d'une course de taxi (`Trip_Price`).")

elif choix == "Car Dataset":
    afficher_modele("Car Dataset - Régression", "car_dataset",
                    description="Prédit le prix de vente d'une voiture (`Selling_Price`).")
