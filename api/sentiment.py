"""
Sentiment Analysis — analyse du sentiment des news financieres.
Utilise yfinance pour recuperer les news et TextBlob pour l'analyse.
"""
import yfinance as yf
from textblob import TextBlob
from datetime import datetime


def get_news(ticker: str, count: int = 10) -> list[dict]:
    """Recupere les dernieres news pour un ticker via yfinance."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return []

        articles = []
        for item in news[:count]:
            articles.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "published": item.get("providerPublishTime", ""),
                "type": item.get("type", ""),
            })
        return articles
    except Exception as e:
        print(f"  [ERREUR] News {ticker}: {e}")
        return []


def analyze_sentiment(text: str) -> dict:
    """Analyse le sentiment d'un texte avec TextBlob."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 (negatif) a +1 (positif)
    subjectivity = blob.sentiment.subjectivity  # 0 (objectif) a 1 (subjectif)

    if polarity > 0.1:
        label = "POSITIF"
    elif polarity < -0.1:
        label = "NEGATIF"
    else:
        label = "NEUTRE"

    return {
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "label": label,
    }


def analyze_ticker_sentiment(ticker: str, count: int = 10) -> dict:
    """Analyse le sentiment global des news d'un ticker."""
    news = get_news(ticker, count)

    if not news:
        return {
            "ticker": ticker,
            "news_count": 0,
            "avg_polarity": 0,
            "sentiment_label": "AUCUNE DONNEE",
            "articles": [],
        }

    analyzed_articles = []
    polarities = []

    for article in news:
        title = article["title"]
        sentiment = analyze_sentiment(title)
        polarities.append(sentiment["polarity"])

        analyzed_articles.append({
            "title": title,
            "publisher": article["publisher"],
            "link": article["link"],
            "sentiment": sentiment["label"],
            "polarity": sentiment["polarity"],
        })

    avg_polarity = sum(polarities) / len(polarities) if polarities else 0

    if avg_polarity > 0.1:
        overall_label = "POSITIF"
    elif avg_polarity < -0.1:
        overall_label = "NEGATIF"
    else:
        overall_label = "NEUTRE"

    # Signal de trading base sur le sentiment
    if avg_polarity > 0.2:
        signal = "ACHAT FORT"
    elif avg_polarity > 0.05:
        signal = "ACHAT"
    elif avg_polarity < -0.2:
        signal = "VENTE FORT"
    elif avg_polarity < -0.05:
        signal = "VENTE"
    else:
        signal = "NEUTRE"

    return {
        "ticker": ticker,
        "news_count": len(analyzed_articles),
        "avg_polarity": round(avg_polarity, 3),
        "sentiment_label": overall_label,
        "signal": signal,
        "analyzed_at": datetime.now().isoformat(),
        "articles": analyzed_articles,
    }


def analyze_multiple_tickers(tickers: list[str]) -> list[dict]:
    """Analyse le sentiment pour plusieurs tickers."""
    results = []
    for ticker in tickers:
        result = analyze_ticker_sentiment(ticker)
        results.append(result)
    return results
