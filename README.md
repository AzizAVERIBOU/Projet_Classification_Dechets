# EcoTri IA — Classification des déchets

Application **Streamlit** et modèle **ResNet50** (transfer learning) pour classer des photos de déchets dans **12 catégories** : batterie, déchet organique, verre (brun / vert / blanc), carton, vêtements, métal, papier, plastique, chaussures, autres déchets.

## Structure du projet

```
Projet_Classification_Dechets/
├── app/
│   ├── app.py                 # Application Streamlit
│   └── .streamlit/
│       └── config.toml        # Configuration Streamlit
├── models/
│   └── modele_tri_dechets_resnet50.keras   # Modèle entraîné (Keras 3 / TF 2.20)
├── notebooks/
│   └── entrainement_resnet50.ipynb         # Notebook d’entraînement
├── data/                      # Non versionné : à remplir localement (voir plus bas)
├── requirements.txt
└── README.md
```

## Prérequis

- **Python 3.10+** (recommandé)
- [TensorFlow 2.20](https://www.tensorflow.org/) (voir `requirements.txt`)

## Installation

Depuis la racine du projet :

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Jeu de données (`data/`)

Le dossier `data/` n’est **pas** inclus dans le dépôt (trop volumineux). Il doit respecter la structure attendue par `image_dataset_from_directory` :

```
data/
├── battery/
├── biological/
├── brown-glass/
├── cardboard/
├── clothes/
├── green-glass/
├── metal/
├── paper/
├── plastic/
├── shoes/
├── trash/
└── white-glass/
```

Le notebook `notebooks/entrainement_resnet50.ipynb` utilise le chemin relatif `../data` depuis `notebooks/`.

## Modèle pré-entraîné

Le fichier `models/modele_tri_dechets_resnet50.keras` est fourni pour exécuter l’appli sans ré-entraîner. Taille ~ **95 Mo** (reste sous la limite GitHub de 100 Mo par fichier ; une version plus lourde pourrait nécessiter [Git LFS](https://git-lfs.github.com/)).

## Lancer l’application

```bash
cd app
streamlit run app.py
```

Puis ouvrir l’URL indiquée dans le terminal (souvent `http://localhost:8501`).

## Entraîner à nouveau

1. Placer les images dans `data/` comme ci-dessus.
2. Ouvrir et exécuter `notebooks/entrainement_resnet50.ipynb`.
3. Exporter le modèle au format Keras dans `models/` et adapter `MODEL_PATH` dans `app/app.py` si le nom du fichier change.

## Licence / crédits

Projet pédagogique — données et modèle à usage non commercial sauf mention contraire des sources des images.
