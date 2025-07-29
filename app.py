import streamlit as st
import pandas as pd
import utils.preprocessing as preprocessing
import utils.sentiment as sentiment
import utils.embeddings as embeddings
import utils.visualization as visualization
import utils.export as export

def main():
    st.title("Plateforme d'analyse sémantique avancée")

    st.markdown("""
        **Chargement du fichier CSV ou Excel avec commentaires, avis, emojis, hashtags.**
        La plateforme analyse en profondeur, génère thèmes, sentiments, visualisations interactives, et exports.
    """)

    uploaded_file = st.file_uploader("Importer fichier CSV ou Excel", type=['csv', 'xlsx'])
    if not uploaded_file:
        st.info("Veuillez importer un fichier pour démarrer l'analyse.")
        return

    # Chargement du fichier en DataFrame
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return

    # Vérification colonne commentaire
    col_commentaire = st.selectbox("Sélectionner la colonne contenant les commentaires/avis", df.columns)
    df = df[[col_commentaire]].rename(columns={col_commentaire: "commentaire"}).dropna(subset=['commentaire'])

    # Prétraitement (nettoyage, lemmatisation, extraction emojis, hashtags)
    df = preprocessing.full_preprocessing(df, text_col='commentaire')

    st.success("Prétraitement terminé ✅")

    # Analyse sentiment
    df = sentiment.analyze_sentiment(df, text_col='texte_nettoye')
    st.success("Analyse sentiment terminée ✅")

    # Clustering thèmes et sous-thèmes
    n_themes = st.slider("Nombre de thèmes principaux", min_value=3, max_value=10, value=5)
    n_subthemes = st.slider("Nombre de sous-thèmes par thème", min_value=2, max_value=6, value=3)
    df, themes, sous_themes = embeddings.cluster_themes(df, text_col='texte_lemmatise', n_themes=n_themes, n_subthemes=n_subthemes)
    st.success("Clustering des thèmes terminé ✅")

    # Affichage résultats principaux
    st.header("Résultats de l'analyse")

    visualization.display_overview(df, themes, sous_themes)
    visualization.display_wordclouds(df)
    visualization.interactive_selection(df)
    # Optionnel : tendance temporelle si colonne date dispo
    if 'date' in df.columns:
        visualization.display_temporal_trends(df, date_col='date')

    # Export des résultats
    st.header("Export des résultats")
    if st.button("Exporter en Excel"):
        excel_bytes = export.to_excel(df)
        st.download_button("Télécharger Excel", data=excel_bytes, file_name="analyse_semantique.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if st.button("Exporter en PDF (rapport)"):
        pdf_bytes = export.to_pdf(df, themes, sous_themes)
        st.download_button("Télécharger PDF", data=pdf_bytes, file_name="rapport_analyse_semantique.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()

