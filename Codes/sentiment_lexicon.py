# sentiment_lexicon.py

class SentimentLexicon:
    """
    Lightweight sentiment lexicon used to guess whether a query
    opinion word is positive, negative, or neutral.
    """

    def __init__(self):
        self.positive_words = set()
        self.negative_words = set()
        self._load_words()

    def _load_words(self):
        print("Loading sentiment lexicon...")

        # --- Positive opinion words ---
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic', 'wonderful',
            'perfect', 'outstanding', 'superb', 'brilliant', 'magnificent', 'marvelous',
            'terrific', 'fabulous', 'impressive', 'remarkable', 'exceptional', 'incredible',
            'beautiful', 'lovely', 'nice', 'pleasant', 'delightful', 'charming', 'attractive',
            'enjoyable', 'satisfying', 'pleasing', 'awesome', 'cool', 'sweet', 'favorite',
            'love', 'loved', 'like', 'liked', 'helpful', 'useful', 'reliable',
            'fast', 'quick', 'responsive', 'smooth', 'easy', 'simple',
            'sharp', 'clear', 'crisp', 'bright', 'vivid', 'rich',
            'sturdy', 'solid', 'durable', 'comfortable', 'quiet',
            'powerful', 'strong', 'stable', 'intuitive', 'convenient',
            'affordable', 'worthwhile', 'valuable', 'efficient',
        }

        # --- Negative opinion words ---
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'pathetic', 'useless',
            'poor', 'worst', 'disappointing', 'frustrating', 'annoying', 'irritating',
            'inferior', 'substandard', 'mediocre', 'inadequate', 'insufficient',
            'unacceptable', 'unreliable', 'broken', 'defective', 'flimsy',
            'noisy', 'loud', 'weak', 'slow', 'sluggish', 'laggy',
            'buggy', 'glitchy', 'confusing', 'complicated', 'unclear',
            'uncomfortable', 'cheap', 'cheesy', 'crappy', 'junk',
            'blurry', 'fuzzy', 'dim', 'dull', 'muted', 'distorted',
            'expensive', 'overpriced', 'waste', 'wasted', 'wasting',
        }
    # Treat emoticon placeholders as sentiment words
        self.positive_words.add("positive_opinion")
        self.negative_words.add("negative_opinion")
    # ---- Public helpers -------------------------------------------------

    def polarity_of_word(self, word: str) -> int:
        """
        Return +1 if word is positive, -1 if negative, 0 if unknown.
        """
        w = word.lower()
        if w in self.positive_words:
            return 1
        if w in self.negative_words:
            return -1
        return 0

    def classify_query_opinion(self, query: str) -> str:
        """
        Roughly decide if the *opinion* part of an aspect:opinion query
        is positive, negative, or neutral.
        """
        if ":" not in query:
            return "neutral"

        _, opinion_part = query.split(":", 1)
        tokens = [t.strip().lower() for t in opinion_part.split() if t.strip()]

        score = sum(self.polarity_of_word(t) for t in tokens)
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"
