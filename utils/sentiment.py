
from transformers import pipeline

# Modèle français multi-classes pour sentiment
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_sentiment(df, text_col='texte_nettoye'):
    def get_sentiment(text):
        if not text or text.strip() == "":
            return {"label": "NEUTRAL", "score": 0.0}
        result = sentiment_analyzer(text[:512])
        label = result[0]['label']
        score = result[0]['score']
        if label in ['1 star', '2 stars']:
            sentiment = 'NEGATIVE'
        elif label == '3 stars':
            sentiment = 'NEUTRAL'
        else:
            sentiment = 'POSITIVE'
        return {"label": sentiment, "score": score}

    sentiments = df[text_col].apply(get_sentiment)
    df['sentiment'] = sentiments.apply(lambda x: x['label'])
    df['sentiment_score'] = sentiments.apply(lambda x: x['score'])
    return df
