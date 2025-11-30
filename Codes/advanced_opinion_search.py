# advanced_opinion_search.py

from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

stemmer = PorterStemmer()
STOP_WORDS = set(nltk.corpus.stopwords.words("english"))


class AdvancedOpinionSearch:
    """
    Advanced method used for Test 4.

    Combines:
      - Aspect/opinion Boolean matching
      - Fuzzy query expansion
      - Rating-based sentiment filter via SentimentLexicon
    """

    def __init__(self, df, lexicon, expander,
                 text_col="filtered_text",
                 rating_col="customer_review_rating"):
        self.df = df
        self.lexicon = lexicon
        self.expander = expander
        self.text_col = text_col
        self.rating_col = rating_col
        self.inverted_index = self._build_index()

    def _build_index(self):
        index = defaultdict(set)
        for doc_id, tokens in self.df[self.text_col].items():
            if not isinstance(tokens, (list, tuple)):
                continue
            for tok in tokens:
                index[tok].add(doc_id)
        print(f"AdvancedOpinionSearch index terms: {len(index)}")
        return index

    def _parse_query(self, query: str):
        if ":" in query:
            aspect_part, opinion_part = query.split(":", 1)
        else:
            aspect_part, opinion_part = query, ""

        def normalize(part):
            raw_terms = [t.strip().lower() for t in part.split()]
            return [
                stemmer.stem(t)
                for t in raw_terms
                if t and t not in STOP_WORDS and len(t) > 2
            ]

        return normalize(aspect_part), normalize(opinion_part)

    def _expand_terms(self, terms, fuzzy_threshold: int):
        expanded = set(terms)
        for t in terms:
            for w in self.expander.expand(t, threshold=fuzzy_threshold):
                expanded.add(w)
        return list(expanded)

    def _sentence_level_and(self, candidate_docs, aspect_terms, opinion_terms):
        """
        Keep only docs where there exists at least one sentence that
        contains:
            - at least one aspect term
            - AND at least one opinion term
        (all compared in stem-space, similar to _parse_query).
        """
        if not aspect_terms or not opinion_terms:
            # If either side is empty, we can't do sentence-level AND
            return candidate_docs

        aspect_set = set(aspect_terms)
        opinion_set = set(opinion_terms)
        kept = set()

        for doc_id in candidate_docs:
            # Use raw review_text for sentence splitting
            try:
                text = self.df.loc[doc_id, "review_text"]
            except KeyError:
                continue

            if not isinstance(text, str):
                continue

            hit = False
            # Split into sentences with NLTK
            for sent in sent_tokenize(text):
                # Light preprocessing: lowercase, tokenize, alpha-only, no stopwords, stem
                toks = [
                    stemmer.stem(tok.lower())
                    for tok in word_tokenize(sent)
                    if tok.isalpha()
                    and tok.lower() not in STOP_WORDS
                    and len(tok) > 2
                ]
                stoks = set(toks)

                if stoks & aspect_set and stoks & opinion_set:
                    hit = True
                    break

            if hit:
                kept.add(doc_id)

        return kept

    def _collect_docs(self, terms, require_all_terms: bool):
        """
        Collect documents for a set of terms.

        - If require_all_terms == False:
              OR logic: docs that contain at least one of the terms.
        - If require_all_terms == True:
              AND logic: docs must contain *all* the terms.
        """
        if not terms:
            return set()

        # OR logic: union of all term docs
        if not require_all_terms:
            docs = set()
            for t in terms:
                docs |= self.inverted_index.get(t, set())
            return docs

        # AND logic: intersection, and fail if any term has 0 docs
        docs = None
        for t in terms:
            term_docs = self.inverted_index.get(t, set())
            if not term_docs:
                # No docs for this term → no doc can satisfy ALL terms
                return set()
            docs = term_docs if docs is None else (docs & term_docs)
            if not docs:
                # Early exit: intersection already empty
                return set()
        return docs

    
    def _apply_rating_filter(self, doc_ids, query_opinion):
        """
        Filter candidate documents based on star rating and the
        query's sentiment (positive/negative/neutral).
        """
        # neutral: keep everything
        if query_opinion == "neutral":
            return set(doc_ids)

        filtered = set()
        for doc_id in doc_ids:
            rating = self.df.loc[doc_id, 'customer_review_rating']
            
            # skip missing / bad ratings
            try:
                rating_val = float(rating)
            except (TypeError, ValueError):
                continue

            if query_opinion == "positive" and rating_val >= 4:
                filtered.add(doc_id)
            elif query_opinion == "negative" and rating_val <= 3:
                filtered.add(doc_id)

        return filtered
    

    def search(self, query: str,
               use_fuzzy_expansion: bool = True,
               use_rating_filter: bool = True,
               fuzzy_threshold: int = 85,
               sentence_and: bool = True,
               require_all_terms: bool = False,):
        """
        Test 4 retrieval:
          - Require aspect AND opinion match (like Test 2)
          - Optionally expand terms with fuzzy matching
          - Optionally require aspect & opinion in the same sentence
          - Optionally filter docs by rating polarity vs query polarity
        """
        # 1) Parse query into stemmed aspect / opinion terms
        aspect_terms, opinion_terms = self._parse_query(query)

        # 2) Fuzzy expansion (if enabled)
        if use_fuzzy_expansion:
            aspect_terms = self._expand_terms(aspect_terms, fuzzy_threshold)
            opinion_terms = self._expand_terms(opinion_terms, fuzzy_threshold)

        # 3) Boolean aspect AND opinion at document level
        
        aspect_docs = self._collect_docs(aspect_terms, require_all_terms)
        opinion_docs = self._collect_docs(opinion_terms, require_all_terms)
        candidate_docs = aspect_docs & opinion_docs

        # 4) OPTIONAL: sentence-level AND (aspect & opinion in same sentence)
        if sentence_and and candidate_docs:
            candidate_docs = self._sentence_level_and(
                candidate_docs, aspect_terms, opinion_terms
            )

        # If nothing left (or rating filter disabled), bail out now
        if not use_rating_filter or not candidate_docs:
            return candidate_docs

        # 5) Rating-based filter using query orientation
        polarity = self.lexicon.classify_query_opinion(query)
        filtered = set()

        for doc_id in candidate_docs:
            rating = self.df.loc[doc_id, self.rating_col]

            # Skip missing / invalid ratings safely
            try:
                rating_val = float(rating)
            except (TypeError, ValueError):
                continue

            if polarity == "positive" and rating_val >= 4:
                filtered.add(doc_id)
            elif polarity == "negative" and rating_val <= 3:
                filtered.add(doc_id)
            elif polarity == "neutral":
                # Neutral: don't constrain by rating
                filtered.add(doc_id)

        return filtered
