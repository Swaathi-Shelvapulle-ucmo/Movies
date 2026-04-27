import re
import string
import numpy as np

class MovieProcessor:
    def __init__(self):
        self.stopwords = {"a", "an", "the", "and", "it", "is", "in", "to", "of", "was", "for", "with", "as"}
        self.topics = {
            "Acting & Cast": ["acting", "actor", "performance", "cast", "lead", "role", "chemistry"],
            "Direction & Visuals": ["director", "cinematography", "visuals", "cgi", "camera", "lighting"],
            "Writing & Plot": ["script", "writing", "plot", "story", "pacing", "dialogue", "screenplay"],
            "Audio & Score": ["music", "soundtrack", "sound", "score", "audio", "voice"]
        }

    def clean_text(self, text):
        text = re.sub(r'<.*?>', '', text)
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        return " ".join([w for w in words if w not in self.stopwords])

    def split_sentences(self, text):
        return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if len(s.strip()) > 5]

    def get_topics(self, text):
        text_lower = text.lower()
        return [t for t, k in self.topics.items() if any(w in text_lower for w in k)] or ["General Impression"]