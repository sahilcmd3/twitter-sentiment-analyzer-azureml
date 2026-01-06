import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import base64
import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class SentimentVisualizer:
    def __init__(self):
        sns.set_style("whitegrid")
        self.colors = {
            "positive": "#2ecc71",
            "negative": "#e74c3c",
            "neutral": "#95a5a6",
        }

    def plot_sentiment_distribution(self, df: pd.DataFrame) -> str:
        """Create pie chart of sentiment distribution"""
        sentiment_counts = df["sentiment"].value_counts()

        plt.figure(figsize=(6, 5))
        colors = [self.colors.get(s, "#95a5a6") for s in sentiment_counts.index]

        plt.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )

        plt.title("Sentiment Distribution", fontsize=16, fontweight="bold")

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        return image_base64

    def plot_sentiment_timeline(self, df: pd.DataFrame) -> str:
        """Create timeline chart"""
        df["date"] = pd.to_datetime(df["created_at"]).dt.date
        timeline_data = df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)

        plt.figure(figsize=(6, 5))

        for sentiment in timeline_data.columns:
            color = self.colors.get(sentiment, "#95a5a6")
            plt.plot(
                timeline_data.index,
                timeline_data[sentiment],
                marker="o",
                label=sentiment.capitalize(),
                color=color,
                linewidth=2,
            )

        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Number of Tweets", fontsize=12)
        plt.title("Sentiment Timeline", fontsize=16, fontweight="bold")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        return image_base64

    def generate_stats(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics"""
        total = len(df)
        counts = df["sentiment"].value_counts().to_dict()

        return {
            "total_tweets": total,
            "positive_count": counts.get("positive", 0),
            "negative_count": counts.get("negative", 0),
            "neutral_count": counts.get("neutral", 0),
            "positive_percentage": (counts.get("positive", 0) / total * 100),
            "negative_percentage": (counts.get("negative", 0) / total * 100),
            "neutral_percentage": (counts.get("neutral", 0) / total * 100),
            "average_confidence": df["confidence"].mean(),
        }
