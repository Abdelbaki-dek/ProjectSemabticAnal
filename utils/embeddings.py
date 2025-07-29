
import torch
from transformers import CamembertTokenizer, CamembertModel
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertModel.from_pretrained("camembert-base")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.cpu().numpy()

def cluster_themes(df, text_col='texte_lemmatise', n_themes=5, n_subthemes=3):
    texts = df[text_col].tolist()
    embeddings = np.array([get_embedding(t) for t in texts])

    kmeans_main = KMeans(n_clusters=n_themes, random_state=42)
    df['theme'] = kmeans_main.fit_predict(embeddings)

    for theme_id in range(n_themes):
        idx_theme = df[df['theme'] == theme_id].index
        emb_theme = embeddings[idx_theme]
        if len(emb_theme) < n_subthemes:
            labels_sub = [0] * len(idx_theme)
        else:
            kmeans_sub = KMeans(n_clusters=n_subthemes, random_state=42)
            labels_sub = kmeans_sub.fit_predict(emb_theme)
        df.loc[idx_theme, 'sous_theme'] = labels_sub
    df['sous_theme'] = df['sous_theme'].astype(int)

    return df, [f"Thème {i}" for i in range(n_themes)], [[f"Sous-thème {j}" for j in range(n_subthemes)] for i in range(n_themes)]
