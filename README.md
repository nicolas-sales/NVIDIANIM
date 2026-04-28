# Nvidia NIM RAG Demo (Streamlit + LangChain)

## Description

Cette application est une **démo de RAG (Retrieval-Augmented Generation)** utilisant :

* **Streamlit** → interface utilisateur
* **LangChain (version moderne)** → orchestration
* **FAISS** → recherche vectorielle
* **NVIDIA NIM (ChatNVIDIA + Embeddings)** → modèle LLM + embeddings
* **PDF Loader** → analyse de documents locaux

L’utilisateur peut poser des questions sur des documents PDF, et l’application répond en se basant uniquement sur leur contenu.

---

## Fonctionnement

1. Chargement des PDF depuis le dossier `./us_census`
2. Découpage en chunks de texte
3. Création des embeddings (NVIDIA)
4. Stockage dans FAISS (vector store)
5. Recherche des passages pertinents (retriever)
6. Injection du contexte dans un prompt
7. Génération de réponse avec le LLM

---

## Installation

### 1. Créer un environnement virtuel

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

### 3. Configurer la clé API NVIDIA

Créer un fichier `.env` :

```
NVIDIA_API_KEY=your_api_key_here
```

---

## ▶Lancer l'application

```bash
streamlit run finalapp.py
```

Puis ouvrir :

```
http://localhost:8501
```

---

## Utilisation

1. Cliquer sur **"Document Embedding"**
2. Attendre la création de la base FAISS
3. Poser une question dans le champ texte
4. Lire la réponse générée
5. Ouvrir **"Document similarity search"** pour voir les sources

---

## Technologies utilisées

* `streamlit`
* `langchain-core`
* `langchain-community`
* `langchain-text-splitters`
* `langchain-nvidia-ai-endpoints`
* `faiss-cpu`
* `pypdf`
* `python-dotenv`

---

# ⚠️ Notes importantes

* L’application utilise `st.session_state` pour éviter de recalculer les embeddings à chaque interaction
* Compatible avec les versions récentes de LangChain (Runnable API)