from pathlib import Path
from typing import cast

import keras
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "modele_tri_dechets_resnet50.keras"
)

st.set_page_config(
    page_title="EcoTri IA — Classification des déchets",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Style global (palette sobre, type “produit” / éco) ---
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {background: transparent;}
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1100px;
        }
        .hero-wrap {
            background: linear-gradient(135deg, #0D7A5F 0%, #0a5c49 50%, #063d32 100%);
            border-radius: 16px;
            padding: 2rem 2.25rem;
            margin-bottom: 1.75rem;
            box-shadow: 0 8px 32px rgba(13, 122, 95, 0.25);
        }
        .hero-title {
            color: #FFFFFF;
            font-size: clamp(1.5rem, 4vw, 2rem);
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0 0 0.5rem 0;
            line-height: 1.2;
        }
        .hero-sub {
            color: rgba(255,255,255,0.88);
            font-size: 1.05rem;
            margin: 0;
            line-height: 1.5;
            max-width: 36rem;
        }
        .badge-row { margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.15);
            color: #fff;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 500;
            border: 1px solid rgba(255,255,255,0.2);
        }
        div[data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 700; }
        .app-footer {
            margin-top: 2.5rem;
            padding: 1.5rem 1.25rem;
            text-align: center;
            border-top: 1px solid rgba(13, 122, 95, 0.18);
            background: rgba(13, 122, 95, 0.05);
            border-radius: 12px;
        }
        .app-footer .footer-name {
            margin: 0 0 0.35rem;
            font-weight: 600;
            color: #1c1c1e;
            font-size: 0.95rem;
        }
        .app-footer .footer-tag {
            margin: 0 0 1rem;
            color: #5f6368;
            font-size: 0.82rem;
            line-height: 1.4;
        }
        .app-footer .footer-links {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
            gap: 0.35rem 0.5rem;
        }
        .app-footer a {
            color: #0D7A5F;
            text-decoration: none;
            font-weight: 500;
            font-size: 0.88rem;
        }
        .app-footer a:hover { text-decoration: underline; color: #063d32; }
        .app-footer .footer-sep { color: #b0b8b6; user-select: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-wrap">
        <h1 class="hero-title">EcoTri IA</h1>
        <p class="hero-sub">Identifiez le type de déchet à partir d’une photo — aide au tri sélectif, propulsé par ResNet50.</p>
        <div class="badge-row">
            <span class="badge">12 catégories</span>
            <span class="badge">ResNet50</span>
            <span class="badge">Prétraitement aligné sur l’entraînement</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

class_names = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass",
]

CATEGORY_FR = {
    "battery": "Piles / batteries",
    "biological": "Déchets organiques",
    "brown-glass": "Verre brun",
    "cardboard": "Carton",
    "clothes": "Textiles",
    "green-glass": "Verre vert",
    "metal": "Métal",
    "paper": "Papier",
    "plastic": "Plastique",
    "shoes": "Chaussures",
    "trash": "Non recyclable / autre",
    "white-glass": "Verre transparent",
}

# Conseil affiché selon la catégorie prédite (repère local type bac bleu / brun — à adapter à votre MRC)
_CONSIGNE_VERRE_PLAST_CONSIGNE = (
    "*Note — verre ou plastique :* s’il s’agit de **contenants consignés**, la meilleure option reste souvent le **retour en magasin**."
)

CONSEIL_TRI_PAR_CATEGORIE: dict[str, str] = {
    "cardboard": (
        "**Bac bleu — recyclables classiques**\n\n"
        "Le **carton** se trie au bac de récupération standard."
    ),
    "paper": (
        "**Bac bleu — recyclables classiques**\n\n"
        "Le **papier** se trie au bac de récupération standard."
    ),
    "plastic": (
        "**Bac bleu — recyclables classiques**\n\n"
        "Le **plastique** se trie au bac de récupération standard.\n\n"
        f"{_CONSIGNE_VERRE_PLAST_CONSIGNE}"
    ),
    "metal": (
        "**Bac bleu — recyclables classiques**\n\n"
        "Le **métal** se trie au bac de récupération standard."
    ),
    "brown-glass": (
        "**Bac bleu — recyclables classiques**\n\n"
        "Le **verre brun** se trie au bac de récupération standard.\n\n"
        f"{_CONSIGNE_VERRE_PLAST_CONSIGNE}"
    ),
    "green-glass": (
        "**Bac bleu — recyclables classiques**\n\n"
        "Le **verre vert** se trie au bac de récupération standard.\n\n"
        f"{_CONSIGNE_VERRE_PLAST_CONSIGNE}"
    ),
    "white-glass": (
        "**Bac bleu — recyclables classiques**\n\n"
        "Le **verre transparent** se trie au bac de récupération standard.\n\n"
        f"{_CONSIGNE_VERRE_PLAST_CONSIGNE}"
    ),
    "biological": (
        "**Bac brun — matières organiques**\n\n"
        "**Déchets organiques** : restes de table, épluchures, marc de café, etc. "
        "Direction le **compost** (ou le bac brun selon les services de votre municipalité)."
    ),
    "battery": (
        "**Collectes spéciales / écocentres**\n\n"
        "**Piles / batteries** : très polluantes, risque d’incendie. "
        "À déposer dans les **points de collecte dédiés** (pharmacies, écocentres, quincailleries). "
        "Ne vont **ni** dans la poubelle **ni** dans le bac bleu."
    ),
    "clothes": (
        "**Réemploi — collecte spéciale**\n\n"
        "**Textiles** : si encore bons, **cloches de dons** (Comptoir familial, Entraide, organismes locaux, etc.) ; "
        "sinon **écocentre** selon les consignes de votre région."
    ),
    "shoes": (
        "**Réemploi — collecte spéciale**\n\n"
        "**Chaussures** : même principe que les vêtements — **boîtes de dons** ou filières de réemploi ; "
        "vérifier auprès de votre municipalité."
    ),
    "trash": (
        "**Déchets ultimes — poubelle**\n\n"
        "**Non recyclable / autre** : à utiliser lorsque les autres options ont été écartées. "
        "Direction le **rebus** (site d’enfouissement ou traitement des résidus selon votre territoire)."
    ),
}

GUIDE_TRI_COMPLET = """
### Recyclables classiques (bac bleu)

Ces matières vont en général au **bac de récupération standard** :

- **cardboard** (carton)
- **paper** (papier)
- **plastic** (plastique)
- **metal** (métal)
- **brown-glass** (verre brun)
- **green-glass** (verre vert)
- **white-glass** (verre transparent)

Pour le **verre** ou le **plastique** : s’il s’agit de **contenants consignés**, la meilleure option reste souvent le **retour en magasin**.

---

### Matières organiques (bac brun)

- **biological** (déchets organiques) : restes de table, épluchures, marc de café, etc. — direction le **compost** / bac brun.

---

### Collectes spéciales / réemploi (écocentres ou organismes)

Ces éléments ont une seconde vie ou sont sensibles ; ils ne vont en principe **ni** dans la poubelle **ni** dans le bac bleu :

- **battery** (piles / batteries) : très polluant, risque d’incendie — **points de collecte dédiés** (pharmacies, écocentres, quincailleries).
- **clothes** (textiles) : **cloches de dons** (Comptoir familial, Entraide, etc.) si l’article est encore bon, sinon **écocentre** selon les consignes locales.
- **shoes** (chaussures) : même idée — **boîtes de dons** / filières de réemploi.

---

### Déchets ultimes (poubelle)

- **trash** (non recyclable / autre) : ce qui reste lorsque toutes les autres options ont été écartées — **rebus** (enfouissement ou traitement des résidus).

---

*Ces repères sont indicatifs : les noms de bacs et les règles **varient selon la municipalité** — vérifiez toujours auprès de votre MRC ou de votre ville.*
"""


@st.cache_resource
def charger_modele():
    return keras.models.load_model(str(MODEL_PATH))


if MODEL_PATH.is_file():
    modele = charger_modele()
else:
    modele = None
    st.error(f"**Modèle introuvable** — `{MODEL_PATH}`")
    st.info(
        "Exécutez la sauvegarde du modèle dans le notebook "
        "`notebooks/entrainement_resnet50.ipynb`, puis relancez l’application."
    )

left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown("##### Importer une image")
    st.caption("Formats JPG, JPEG ou PNG — photo nette et bien éclairée pour de meilleurs résultats.")
    fichier_upload = st.file_uploader(
        "Glisser-déposer ou parcourir les fichiers",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

with right:
    st.markdown("##### État du service")
    if modele is not None:
        st.success("Modèle chargé et prêt à l’analyse.")
    else:
        st.warning("En attente du fichier modèle `.keras`.")

st.divider()

if fichier_upload is not None:
    if modele is None:
        st.warning("Impossible d’analyser l’image tant que le modèle est absent.")
    else:
        assert modele is not None
        model_actif = cast(keras.Model, modele)
        # PNG / images avec canal alpha → RGB obligatoire (le modèle attend 3 canaux, pas 4)
        image = Image.open(fichier_upload).convert("RGB")
        img_col, res_col = st.columns([1, 1.1], gap="large")

        with img_col:
            st.markdown("##### Aperçu")
            st.image(
                image,
                caption="Image fournie",
                use_container_width=True,
            )

        with res_col:
            st.markdown("##### Résultat de l’analyse")
            with st.spinner("Analyse de l’image en cours…"):
                image_redimensionnee = image.resize((224, 224))
                img_array = keras.preprocessing.image.img_to_array(image_redimensionnee)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = keras.applications.resnet50.preprocess_input(img_array)
                out = model_actif(img_array, training=False)
                predictions = np.asarray(out)

            probs = predictions[0]
            index_prediction = int(np.argmax(probs))
            classe_predite = class_names[index_prediction]
            confiance = float(np.max(probs) * 100)
            libelle = CATEGORY_FR.get(classe_predite, classe_predite)

            st.metric(
                label="Catégorie prédite",
                value=libelle,
                delta=f"{confiance:.1f} % de confiance",
                delta_color="off",
            )

            st.progress(min(confiance / 100.0, 1.0), text="Niveau de confiance")

            order = np.argsort(probs)[::-1]
            if len(order) >= 2:
                st.markdown("**Autres hypothèses probables**")
                i2 = int(order[1])
                if len(order) >= 3:
                    h2, h3 = st.columns(2)
                    i3 = int(order[2])
                    with h2:
                        st.metric(
                            "2ᵉ choix",
                            CATEGORY_FR[class_names[i2]],
                            f"{float(probs[i2] * 100):.1f} %",
                            delta_color="off",
                        )
                    with h3:
                        st.metric(
                            "3ᵉ choix",
                            CATEGORY_FR[class_names[i3]],
                            f"{float(probs[i3] * 100):.1f} %",
                            delta_color="off",
                        )
                else:
                    st.metric(
                        "2ᵉ choix",
                        CATEGORY_FR[class_names[i2]],
                        f"{float(probs[i2] * 100):.1f} %",
                        delta_color="off",
                    )

            with st.container(border=True):
                st.markdown("##### Que faire de ce déchet ?")
                st.markdown(CONSEIL_TRI_PAR_CATEGORIE.get(classe_predite, ""))

        st.divider()
        st.caption(
            "Les prédictions de l’IA sont indicatives : croisez toujours avec les consignes officielles de votre municipalité."
        )

elif modele is not None:
    st.info("Choisissez une image dans la colonne de gauche pour lancer la classification.")

with st.expander("Guide de tri — les 12 catégories (référence)", expanded=False):
    st.markdown(GUIDE_TRI_COMPLET)

st.divider()
st.markdown(
    """
    <div class="app-footer">
        <p class="footer-name">Aziz AVERIBOU</p>
        <p class="footer-tag">Développement &amp; projets — n’hésitez pas à me contacter ou à parcourir mon travail.</p>
        <div class="footer-links">
            <a href="https://github.com/AzizAVERIBOU" target="_blank" rel="noopener noreferrer">GitHub</a>
            <span class="footer-sep">·</span>
            <a href="https://www.linkedin.com/in/aziz-averibou-51b782323/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
            <span class="footer-sep">·</span>
            <a href="https://profil-aziz.vercel.app/" target="_blank" rel="noopener noreferrer">Portfolio</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
