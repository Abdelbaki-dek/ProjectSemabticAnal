
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

def display_overview(df, themes, sous_themes):
    st.subheader("Répartition des sentiments")
    pie_data = df['sentiment'].value_counts().reset_index()
    pie_data.columns = ['sentiment', 'count']
    fig_pie = px.pie(pie_data, values='count', names='sentiment',
                     color='sentiment',
                     color_discrete_map={'POSITIVE': 'green', 'NEUTRAL': 'gray', 'NEGATIVE': 'red'},
                     title="Répartition sentimentale")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Répartition des thèmes")
    fig_theme = px.histogram(df, x='theme', title="Thèmes principaux")
    st.plotly_chart(fig_theme, use_container_width=True)

    st.subheader("Répartition des sous-thèmes")
    fig_subtheme = px.histogram(df, x='sous_theme', title="Sous-thèmes")
    st.plotly_chart(fig_subtheme, use_container_width=True)

    st.subheader("Heatmap : Thèmes vs Sentiments")
    heat_data = pd.crosstab(df['theme'], df['sentiment'])
    fig_heat = px.imshow(heat_data, text_auto=True, color_continuous_scale='Viridis',
                         labels=dict(x="Sentiment", y="Thème", color="Nombre d'avis"))
    st.plotly_chart(fig_heat, use_container_width=True)

def _wordcloud_figure(freq_dict, title="Nuage de mots"):
    wordcloud = WordCloud(width=600, height=400, background_color="white").generate_from_frequencies(freq_dict)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title)
    return fig

def display_wordclouds(df):
    st.subheader("Nuage des mots clés")
    all_lemmas = [lemma for sublist in df['lemmas'] for lemma in sublist]
    freq_lemmas = pd.Series(all_lemmas).value_counts()
    fig = _wordcloud_figure(freq_lemmas.to_dict(), "Nuage des mots clés")
    st.pyplot(fig)

    st.subheader("Nuage des hashtags")
    all_hashtags = [tag.lower() for sublist in df['hashtags'] for tag in sublist]
    freq_tags = pd.Series(all_hashtags).value_counts()
    fig = _wordcloud_figure(freq_tags.to_dict(), "Nuage des hashtags")
    st.pyplot(fig)

    st.subheader("Nuage des emojis")
    all_emojis = [emoji for sublist in df['emojis'] for emoji in sublist]
    freq_emojis = pd.Series(all_emojis).value_counts()
    fig = _wordcloud_figure(freq_emojis.to_dict(), "Nuage des emojis")
    st.pyplot(fig)

def interactive_selection(df):
    st.subheader("Exploration interactive des mots, hashtags et emojis")

    freq_lemmas = pd.Series([lemma for sublist in df['lemmas'] for lemma in sublist]).value_counts()
    freq_tags = pd.Series([tag.lower() for sublist in df['hashtags'] for tag in sublist]).value_counts()
    freq_emojis = pd.Series([emoji for sublist in df['emojis'] for emoji in sublist]).value_counts()

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_word = st.selectbox("Mots clés fréquents", freq_lemmas.index[:100])
    with col2:
        selected_tag = st.selectbox("Hashtags fréquents", freq_tags.index[:100])
    with col3:
        selected_emoji = st.selectbox("Emojis fréquents", freq_emojis.index[:100])

    def show_examples(word, col_name):
        st.write(f"Exemples d'avis contenant : '{word}'")
        filt = df[df[col_name].apply(lambda x: word in x if isinstance(x, list) else False)]
        for idx, row in filt.head(15).iterrows():
            st.markdown(f"- {row.get('commentaire', '---')}")

    if selected_word:
        show_examples(selected_word, 'lemmas')
    if selected_tag:
        show_examples(selected_tag, 'hashtags')
    if selected_emoji:
        show_examples(selected_emoji, 'emojis')

def display_temporal_trends(df, date_col='date'):
    if date_col not in df.columns:
        return
    st.subheader("Tendances temporelles des sentiments")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df_time = df.dropna(subset=[date_col])
    if df_time.empty:
        st.write("Aucune donnée temporelle valide.")
        return

    df_time['date_only'] = df_time[date_col].dt.date
    trend_df = df_time.groupby(['date_only', 'sentiment']).size().reset_index(name='count')
    fig = px.line(trend_df, x='date_only', y='count', color='sentiment', 
                  title="Évolution des sentiments dans le temps")
    st.plotly_chart(fig, use_container_width=True)
