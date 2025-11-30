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
            'fine', 'solid', 'reliable', 'dependable', 'trustworthy', 'quality', 'premium',
            'superior', 'best', 'top', 'first-class', 'high-quality', 'well-made',
            'strong', 'powerful', 'effective', 'efficient', 'fast', 'quick', 'speedy',
            'smooth', 'easy', 'simple', 'convenient', 'comfortable', 'cozy', 'relaxing',
            'satisfying', 'pleased', 'happy', 'satisfied', 'content', 'delighted', 'thrilled',
            'love', 'like', 'enjoy', 'appreciate', 'recommend', 'useful', 'helpful',
            'valuable', 'worthwhile', 'beneficial', 'advantageous', 'clear', 'sharp',
            'bright', 'vivid', 'crisp', 'clean', 'fresh', 'new', 'modern', 'stylish',
            'elegant', 'sleek', 'compact', 'portable', 'lightweight', 'durable', 'sturdy'
        }

        # --- Negative opinion words ---
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'pathetic', 'useless',
            'worthless', 'disappointing', 'frustrating', 'annoying', 'irritating', 'infuriating',
            'poor', 'inferior', 'substandard', 'mediocre', 'inadequate', 'insufficient',
            'defective', 'faulty', 'broken', 'damaged', 'flawed', 'imperfect', 'problematic',
            'unreliable', 'unstable', 'inconsistent', 'unpredictable', 'buggy', 'glitchy',
            'slow', 'sluggish', 'laggy', 'delayed', 'unresponsive', 'frozen', 'stuck',
            'difficult', 'hard', 'complicated', 'complex', 'confusing', 'unclear', 'ambiguous',
            'uncomfortable', 'inconvenient', 'awkward', 'clumsy', 'bulky', 'heavy', 'loud',
            'noisy', 'harsh', 'rough', 'cheap', 'flimsy', 'fragile', 'weak', 'thin',
            'small', 'tiny', 'limited', 'restricted', 'narrow', 'short', 'brief', 'quick',
            'waste', 'money', 'time', 'effort', 'regret', 'mistake', 'error', 'fail',
            'failure', 'disaster', 'nightmare', 'mess', 'junk', 'garbage', 'trash',
            'hate', 'dislike', 'avoid', 'skip', 'ignore', 'return', 'refund', 'replace',
            'issues', 'problems', 'troubles', 'difficulties', 'challenges', 'concerns',
            'complaints', 'criticisms', 'faults', 'drawbacks', 'disadvantages', 'downsides',
            'blur', 'blurry', 'fuzzy', 'dim', 'dark', 'faded', 'dull', 'muted', 'distorted'
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
