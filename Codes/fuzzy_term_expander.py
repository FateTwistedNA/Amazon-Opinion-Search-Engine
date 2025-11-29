# fuzzy_term_expander.py

from collections import Counter
from fuzzywuzzy import process, fuzz


class FuzzyTermExpander:
    """
    Builds a vocabulary of frequent tokens and supports fuzzy expansion
    of query terms using fuzzywuzzy.
    """

    def __init__(self, df, text_col="filtered_text", min_freq=15):
        self.df = df
        self.text_col = text_col
        self.min_freq = min_freq
        self.vocab = []
        self._build_vocab()

    def _build_vocab(self):
        counts = Counter()
        for tokens in self.df[self.text_col]:
            if not isinstance(tokens, (list, tuple)):
                continue
            for tok in tokens:
                counts[tok] += 1

        self.vocab = [w for w, c in counts.items() if c >= self.min_freq]
        print(f"FuzzyTermExpander vocab size (freq >= {self.min_freq}): {len(self.vocab)}")

    def expand(self, term: str, threshold: int = 80, limit: int = 5):
        """
        Return a list of similar tokens in the vocabulary.
        """
        if not self.vocab:
            return []

        matches = process.extract(term, self.vocab, scorer=fuzz.ratio, limit=limit)
        return [w for (w, score) in matches if score >= threshold]
