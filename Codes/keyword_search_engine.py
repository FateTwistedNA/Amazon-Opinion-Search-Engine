# keyword_search_engine.py

from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
STOP_WORDS = set(nltk.corpus.stopwords.words("english"))


class KeywordSearchEngine:

    # Baseline keyword-based Boolean search over filtered_text.
    # Test 1: aspect-only retrieval
    # Test 2: aspect AND opinion
    # Test 3: aspect OR opinion

    def __init__(self, df, lexicon, text_col="filtered_text", rating_col="customer_review_rating"):
        self.df = df
        self.lexicon = lexicon
        self.text_col = text_col
        self.rating_col = rating_col
        self.inverted_index = self._build_inverted_index()

    def _build_inverted_index(self):
        print("Building keyword inverted index...")
        index = defaultdict(set)

        for doc_id, tokens in self.df[self.text_col].items():
            if not isinstance(tokens, (list, tuple)):
                continue
            for tok in tokens:
                index[tok].add(doc_id)

        print(f"Inverted index terms: {len(index)}")
        return index

    def _parse_query(self, query: str):

        # Normalize (lowercase + stemming + stopword filtering) and parse aspect1 aspect2:opinion1 opinion2 into aspect_terms, opinion_terms

        if ":" in query:
            aspect_part, opinion_part = query.split(":", 1)
        else:
            aspect_part, opinion_part = query, ""

        def normalize(part):
            raw_terms = [t.strip().lower() for t in part.split()]
            terms = [
                stemmer.stem(t)
                for t in raw_terms
                if t and t not in STOP_WORDS and len(t) > 2
            ]
            return terms

        aspect_terms = normalize(aspect_part)
        opinion_terms = normalize(opinion_part)
        return aspect_terms, opinion_terms

    # ------------------------------------------------------------------ #
    #  Test 1 / 2 / 3  (Baseline)
    # ------------------------------------------------------------------ #

    def retrieve_aspect_only(self, query: str):

        # Test 1 – Aspect-only Boolean retrieval.
        # Return docs containing at least one aspect term.

        aspect_terms, _ = self._parse_query(query)

        docs = set()
        for term in aspect_terms:
            docs |= self.inverted_index.get(term, set())
        return docs

    def retrieve_aspect_and_opinion(self, query: str):

        # Test 2 – Boolean Aspect AND Opinion match.
        # (>=1 aspect term) AND (>=1 opinion term).

        aspect_terms, opinion_terms = self._parse_query(query)
        # sanity-check query parsing
        # print("DEBUG T2:", query, "->", aspect_terms, opinion_terms)
        
        # docs containing any aspect term
        aspect_docs = set()
        for term in aspect_terms:
            aspect_docs |= self.inverted_index.get(term, set())

        # docs containing any opinion term
        opinion_docs = set()
        for term in opinion_terms:
            opinion_docs |= self.inverted_index.get(term, set())

        return aspect_docs & opinion_docs

    def retrieve_aspect_or_opinion(self, query: str):

        # Test 3 – Aspect OR Opinion match.
        # At least one aspect or one opinion term.

        aspect_terms, opinion_terms = self._parse_query(query)

        docs = set()
        for term in aspect_terms + opinion_terms:
            docs |= self.inverted_index.get(term, set())
        return docs
