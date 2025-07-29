
import re
import pandas as pd
import spacy
import emoji

nlp = spacy.load("fr_core_news_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # URL
    text = re.sub(r"@\w+", "", text)     # mentions
    text = re.sub(r"#[\w\-]+", "", text) # hashtags retirés ici (on extrait à part)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)  # enlever ponctuation sauf emojis (capturés avant)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_emojis(text):
    return [c for c in text if c in emoji.UNICODE_EMOJI['en']]

def extract_hashtags(text):
    return re.findall(r"#(\w+)", text)

def lemmatize_text(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_.strip()]
    return lemmas

def full_preprocessing(df, text_col='commentaire'):
    df['emojis'] = df[text_col].apply(extract_emojis)
    df['hashtags'] = df[text_col].apply(extract_hashtags)
    df['texte_nettoye'] = df[text_col].apply(clean_text)
    df['texte_lemmatise'] = df['texte_nettoye'].apply(lambda x: " ".join(lemmatize_text(x)))
    df['lemmas'] = df['texte_nettoye'].apply(lemmatize_text)
    return df
