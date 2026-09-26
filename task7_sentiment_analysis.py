import pandas as pd
import numpy as np

print(" --- Step 1: Ingesting Raw Text Feedback Streams ----")
feedback_records = {
    "Intern_Name": ["Samra", "Ahmed", "Sana", "Zain", "Ali"],
    "Review_Text": [
        "This Data analytics program is amazing and I learned a lot!",
        "The internship tasks were updated on Saturday afternoon.",
        "The system dashboard interface is too confusing and slow.",
        "Python coding exercise are great fun and highly practical.",
        "I submitted my performance excel sheet on time yesterday."


    ]

}
df = pd.DataFrame(feedback_records)

print("\nIntial Unprocessed Text Panels:")
print(df)

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

print("\n--- Step 2: Downloading VADER Lexicon Sub_Systems ---")
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

print("\n--- Step 3: Running Emotional Sentiment Analysis Queries ---")
df["Scores"] = df["Reciew_Text"].apply(lambda text: sia.polarity_scores(text)["compound"])
df["Sentiment_Result"] = df["Scores"].apply(lambda score: "POSTIVE" if score > 0.05 else ("NEGTIVE" if score < -0.05 else "NEUTRAL"))
print("\n--- Final Text Auditing Performance Dashboard ---")
print(df[["Intern_Name", "Sentiment_Result"]])
