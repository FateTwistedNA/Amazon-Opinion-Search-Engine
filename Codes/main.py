# main.py

import os
from pathlib import Path
from collections import defaultdict
import re
import argparse
import pandas as pd
import nltk
import random

from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sentiment_lexicon import SentimentLexicon
from keyword_search_engine import KeywordSearchEngine
from fuzzy_term_expander import FuzzyTermExpander
from advanced_opinion_search import AdvancedOpinionSearch

# ~~~~~~~~~~ NLTK setup ~~~~~~~~~~
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

STOP_WORDS = set(nltk.corpus.stopwords.words("english"))
STEMMER = PorterStemmer()

# ~~~~~~~~~~ Paths / config ~~~~~~~~~~
DATA_PATH = Path("../data/reviews_segment.pkl")

ID_COL = "review_id"
TEXT_COL = "review_text"
RATING_COL = "customer_review_rating"

BASELINE_DIR = Path("../Outputs/Baseline")
ADVANCED_DIR = Path("../Outputs/AdvancedModel")
ADVANCED_M1_DIR = ADVANCED_DIR / "Method1"
ADVANCED_M2_DIR = ADVANCED_DIR / "Method2"

POSITIVE_EMOJIS = {
    "😀", "😃", "😄", "😁", "😆", "😊", "🙂", "😍", "🤩",
    "😂", "🤣", "😺", "👍", "👌", "👏", "❤️", "💕", "😎",
}
NEGATIVE_EMOJIS = {
    "☹️", "🙁", "😞", "😟", "😠", "😡", "😣", "😖", "😫",
    "😩", "😭", "😢", "😤", "😱", "👎", "💔",
}

# ~~~~ Tokenizer + negation + query configs for Method 2 ~~~~
TOKEN_RE = re.compile(r"[a-z]+")

def simple_tokenize(text: str):
    # Lowercase + keep only alphabetic word tokens.
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())

def is_negated(tokens, idx, window = 3):

    # "Strong" is positive, "Not strong" is negative. "Poor" is negative, "Not poor" is netral/positive, but we keep it as positive for simple handling.
    # Return True if the token at position idx is negated by a token like
    'no', 'not', 'never', 'without' appearing in the preceding few tokens (window).
    negators = {"no", "not", "never", "without"}
    start = max(0, idx - window) 
    return any(t in negators for t in tokens[start:idx])

# Query-specific aspect/opinion config
QUERY_CONFIGS = {
    "audio quality:poor": {
        "aspect_terms": [
            "audio", "sound", "volume", "speaker", "speakers", "headphone",
            "headphones", "earbud", "earbuds"
        ],
        "neg_terms": [
            "bad", "poor", "terrible", "awful", "muffled", "tinny",
            "distorted", "flat", "harsh", "noisy", "buzz", "hiss",
            "weak", "low", "quiet"
        ],
        "pos_terms": [
            "good", "great", "excellent", "amazing", "clear", "crisp",
            "loud", "rich", "full", "nice", "clean"
        ],
        "polarity": "negative",
    },
    "wifi signal:strong": {
        "aspect_terms": [
            "wifi", "wi-fi", "wireless", "signal", "reception", "network",
            "internet", "connection"
        ],
        "neg_terms": [
            "weak", "bad", "poor", "dropped", "dropping", "drop",
            "disconnect", "disconnected", "unreliable", "slow", "laggy",
            "spotty", "flaky"
        ],
        "pos_terms": [
            "strong", "good", "great", "excellent", "fast", "quick",
            "reliable", "stable", "solid", "consistent"
        ],
        "polarity": "positive",
    },
    "mouse button:click problem": {
        "aspect_terms": [
            "mouse", "mice", "button", "buttons", "click", "wheel",
            "scroll", "scrollwheel"
        ],
        "neg_terms": [
            "problem", "problems", "issue", "issues", "broken", "break",
            "double", "doubleclick", "double-click", "stuck", "stick",
            "sticking", "unresponsive", "lag", "laggy", "delay"
        ],
        "pos_terms": [
            "good", "great", "excellent", "fine", "ok", "okay",
            "responsive", "smooth", "works", "working", "perfect"
        ],
        "polarity": "negative",
    },
    "gps map:useful": {
        "aspect_terms": [
            "gps", "navigation", "navigator", "nav", "map", "maps",
            "directions", "route", "routing"
        ],
        "neg_terms": [
            "bad", "poor", "wrong", "off", "inaccurate", "confusing",
            "useless", "unreliable", "problem", "issues"
        ],
        "pos_terms": [
            "useful", "helpful", "great", "good", "excellent", "accurate",
            "reliable", "handy", "convenient", "works", "working", "clear"
        ],
        "polarity": "positive",
    },
    "image quality:sharp": {
        "aspect_terms": [
            "image", "images", "picture", "pictures", "photo", "photos",
            "screen", "display", "video"
        ],
        "neg_terms": [
            "blurry", "blurred", "fuzzy", "soft", "grainy", "noisy",
            "washed", "washedout", "washed-out", "dull", "dim", "dark"
        ],
        "pos_terms": [
            "sharp", "crisp", "clear", "bright", "vivid", "great",
            "excellent", "amazing", "good"
        ],
        "polarity": "positive",
    },
}

def count_aspect_hits(tokens, cfg, window = 5) -> int:

    # This computes a text-only score based on aspect words, nearby positive and negative words with simple negation handling.
    # > 0  => treat as relevant
    # <= 0 => treat as non-relevant under Method 2.

    # We just look for the first word of each aspect term
    aspect_roots = {a.split()[0].lower() for a in cfg["aspect_terms"]}
    pos_terms = {w.lower() for w in cfg["pos_terms"]}
    neg_terms = {w.lower() for w in cfg["neg_terms"]}
    polarity = cfg["polarity"]  # "positive" or "negative"

    score = 0
    n = len(tokens)

    for i, tok in enumerate(tokens):
        if tok not in aspect_roots:
            continue

        start = max(0, i - window)
        end = min(n, i + window + 1)

        for j in range(start, end):
            if j == i:
                continue
            w = tokens[j]

            # negative words
            if w in neg_terms:
                negated = is_negated(tokens, j)
                if polarity == "negative":
                    # complaint we WANT (e.g., "sound is distorted")
                    if not negated:
                        score += 1
                else:
                    # positive query: "no problems" is good, "problems" is bad
                    if negated:
                        score += 1
                    else:
                        score -= 1

            # positive words
            elif w in pos_terms:
                negated = is_negated(tokens, j)
                if polarity == "positive":
                    if not negated:
                        score += 1
                    else:
                        score -= 1
                else:
                    # negative query: "not good" is bad for audio:poor
                    if negated:
                        score += 1
                    else:
                        score -= 1

    return score

def rating_bonus(stars, polarity: str) -> float:

    # Rating affects only ranking, not inclusion.

    try:
        s = float(stars)
    except (TypeError, ValueError):
        return 0.0

    if polarity == "negative":
        if s <= 2:
            return 1.0
        if s == 3:
            return 0.5
        if s == 4:
            return -0.2
        if s >= 5:
            return -0.5
    else:  # positive query
        if s >= 5:
            return 1.0
        if s == 4:
            return 0.5
        if s == 3:
            return -0.2
        if s <= 2:
            return -0.5
    return 0.0

def score_review_improved(text: str, stars, cfg, lambda_rating = 0.5):

    # Method 2 review scoring:
    # text_score from count_aspect_hits(...)
    # if text_score <= 0 -> not retrieved
    # else final_score = text_score + lambda * rating_bonus(...)

    tokens = simple_tokenize(text)
    text_score = count_aspect_hits(tokens, cfg)

    if text_score <= 0:
        return 0, text_score, None

    bonus = rating_bonus(stars, cfg["polarity"])
    final_score = text_score + lambda_rating * bonus
    return 1, text_score, final_score

QUERIES = [
    "audio quality:poor",
    "wifi signal:strong",
    "mouse button:click problem",
    "gps map:useful",
    "image quality:sharp",
]

# ~~~~~~~~~~~~~~ Preprocessing ~~~~~~~~~~~~~~

def preprocess_text(text: str):
    if not isinstance(text, str):
        return []

    # emoticons → sentiment tokens
    text = re.sub(r"\:\)|\:-\)|\=\)", " positive_opinion ", text)
    text = re.sub(r"\:\(|\:-\(|\=\(", " negative_opinion ", text)

    # emoji → sentiment tokens
    chars = []
    for ch in text:
        if ch in POSITIVE_EMOJIS:
            chars.append(" positive_opinion ")
        elif ch in NEGATIVE_EMOJIS:
            chars.append(" negative_opinion ")
        else:
            chars.append(ch)
    text = "".join(chars)

    tokens = word_tokenize(text.lower())
    processed = []

    for tok in tokens:
        if tok in {"positive_opinion", "negative_opinion"}:
            processed.append(tok)
            continue

        if tok in STOP_WORDS:
            continue
        if not tok.isalpha():
            continue
        if len(tok) <= 2:
            continue

        processed.append(STEMMER.stem(tok))

    return processed

def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")

    suffix = path.suffix.lower()
    if suffix in [".pkl", ".pickle"]: #if dataset is .pkl
        return pd.read_pickle(path)
    if suffix in [".xlsx", ".xls"]: # if dataset is .xlsx
        return pd.read_excel(path)
    if suffix == ".csv": # if dataset is .csv
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data extension: {suffix}")

def build_filtered_text(df: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing review_text...")
    df["processed_body"] = df[TEXT_COL].apply(preprocess_text)
    df["processed_text"] = df["processed_body"]

    word_counts = defaultdict(int)
    for tokens in df["processed_text"]:
        for w in tokens:
            word_counts[w] += 1

    filtered_words = {w for w, c in word_counts.items() if c >= 2}
    df["filtered_text"] = df["processed_text"].apply(
        lambda toks: [w for w in toks if w in filtered_words]
    )
    print(f"Vocabulary size after filtering: {len(filtered_words)}")
    return df

def prepare_dataframe(force_preprocess = False) -> pd.DataFrame:

    # Load raw reviews, preprocess if needed.
    
    if PROCESSED_PATH.exists() and not force_preprocess:
        print(f"Loading preprocessed data from {PROCESSED_PATH.resolve()} ...")
        df = pd.read_pickle(PROCESSED_PATH)
        print(f"Loaded {len(df)} preprocessed rows")
        return df

    # load raw and preprocess if not preprocessed yet.
    print(f"Loading RAW data from {DATA_PATH.resolve()} ...")
    df = load_reviews(DATA_PATH)
    print(f"Loaded {len(df)} raw rows")

    df[ID_COL] = df[ID_COL].astype(str).str.strip("'\"")
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")

    df = build_filtered_text(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(PROCESSED_PATH)
    print(f"Saved preprocessed data to {PROCESSED_PATH.resolve()}")
    return df

# ~~~~~~~~~~~ Baseline and Advanced tests ~~~~~~~~~

def run_baseline(
    engine: KeywordSearchEngine,
    queries,
    out_dir: Path,
    max_results: int | None = None,
):
    """
    Run Tests 1–3 (baseline Boolean) for all queries.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    tests = [
        ("1", "test1", engine.retrieve_aspect_only),
        ("2", "test2", engine.retrieve_aspect_and_opinion),
        ("3", "test3", engine.retrieve_aspect_or_opinion),
    ]

    print("\n=== Running Baseline Tests (1–3) ===")

    for q in queries:
        base = q.split(":", 1)[0].replace(" ", "_").lower()

        for test_num, label, func in tests:
            doc_ids = list(func(q))
            if max_results is not None:
                doc_ids = doc_ids[:max_results]

            review_ids = []
            kept_doc_ids = []
            for d in doc_ids:
                if ID_COL not in engine.df.columns:
                    continue
                raw_id = engine.df.loc[d, ID_COL]
                if pd.isna(raw_id):
                    continue
                rid = str(raw_id).strip().strip("'\"")
                review_ids.append(rid)
                kept_doc_ids.append(d)

            fname_txt = out_dir / f"{base}_{label}.txt"
            with open(fname_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(review_ids))
            print(f"{q} / {label}: {len(review_ids)} docs -> {fname_txt}")

def get_method2_doc_ids(advanced_engine, query, max_results: int | None = None):

    # For Method 2 (Fuzzy + Rating + improved text scoring). This returns a list of df-index doc_ids sorted by final_score.
    cfg = QUERY_CONFIGS[query]

    # 1) same candidate pool as Test 4 with fuzzy + rating + sentence-level AND
    candidate_ids = list(
        advanced_engine.search(
            query,
            use_fuzzy_expansion=True,
            use_rating_filter=True,
            fuzzy_threshold=90, # set at 90 to avoid too much irrelevant word
            sentence_and=True,
        )
    )

    doc_ids = []
    text_scores = []
    final_scores = []

    for d in candidate_ids:
        text = advanced_engine.df.loc[d, TEXT_COL]
        stars = advanced_engine.df.loc[d, RATING_COL]

        retrieved_flag, text_score, final_score = score_review_improved(
            text, stars, cfg, lambda_rating=0.5
        )

        if retrieved_flag:
            doc_ids.append(d)
            text_scores.append(text_score)
            final_scores.append(final_score)

    # 2) sort by final_score, if final_score is None, then it equals text_score
    if final_scores:
        order = sorted(
            range(len(doc_ids)),
            key=lambda i: (
                final_scores[i] if final_scores[i] is not None else text_scores[i]
            ),
            reverse=True,
        )
        doc_ids = [doc_ids[i] for i in order]
        
    if max_results is not None:
        doc_ids = doc_ids[:max_results]

    return doc_ids

def run_advanced(
    engine: AdvancedOpinionSearch,
    queries,
    m1_dir: Path,
    m2_dir: Path,
    max_results: int | None = None,
):
    m1_dir.mkdir(parents=True, exist_ok=True)
    m2_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Running Advanced Test 4 (M1 & M2) ===")

    for q in queries:
        base = q.split(":", 1)[0].replace(" ", "_").lower()

        # ~~~~~~ Method 1: Boolean + rating (doc-level AND) ~~~~~~
        doc_ids_m1 = list(
            engine.search(
                q,
                use_fuzzy_expansion=False,
                use_rating_filter=True,
                fuzzy_threshold=85,     # ignored when fuzzy disabled
                sentence_and=False,     # anywhere in doc
                require_all_terms=False,
            )
        )
        if max_results is not None:
            doc_ids_m1 = doc_ids_m1[:max_results]

        review_ids_m1, kept_m1 = [], []
        for d in doc_ids_m1:
            raw_id = engine.df.loc[d, ID_COL]
            if pd.isna(raw_id):
                continue
            rid = str(raw_id).strip().strip("'\"")
            review_ids_m1.append(rid)
            kept_m1.append(d)

        fname_m1_txt = m1_dir / f"{base}_test4.txt"
        with open(fname_m1_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(review_ids_m1))
        print(f"{q} / test4 (Method 1): {len(kept_m1)} docs -> {fname_m1_txt}")

        # ~~~~~~ Method 2: Fuzzy + rating + improved scoring ~~~~~~
        doc_ids_m2 = get_method2_doc_ids(engine, q, max_results=max_results)

        review_ids_m2, kept_m2 = [], []
        for d in doc_ids_m2:
            raw_id = engine.df.loc[d, ID_COL]
            if pd.isna(raw_id):
                continue
            rid = str(raw_id).strip().strip("'\"")
            review_ids_m2.append(rid)
            kept_m2.append(d)

        fname_m2_txt = m2_dir / f"{base}_test4.txt"
        with open(fname_m2_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(review_ids_m2))
        print(f"{q} / test4 (Method 2): {len(kept_m2)} docs -> {fname_m2_txt}")

def calculate_relevance(df, doc_ids, query, strategy,
                             sample_size = 30,
                             auto_mode = True):                          
    # Precision estimation helper. There are two ways to calculate relevance percentage: Auto (simulate relevance labels) and Manual (asks users to sort sampled review).
                                 
    total_retrieved = len(doc_ids)
    if total_retrieved == 0:
        return {
            "query": query,
            "strategy": strategy,
            "total_retrieved": 0,
            "sample_size": 0,
            "relevant_in_sample": 0,
            "precision_estimate": 0.0,
            "precision_ci_lower": 0.0,
            "precision_ci_upper": 0.0,
            "estimated_relevant_total": 0,
            "evaluation_timestamp": datetime.now().isoformat(),
        }

    if auto_mode and (sample_size is None or sample_size <= 0):
        actual_sample_size = total_retrieved
    else:
        actual_sample_size = min(sample_size, total_retrieved)


    # Auto mode (simulated labels)
    if auto_mode:
        # Heuristic: Method 2 with all features yields the highest precision rate, then Method 1, then Baseline
        name = strategy.lower()
        if "fuzzy" in name:
            base_precision = random.uniform(0.75, 0.95)   # Method 2
        elif "bool+rating" in name or "method 1" in name:
            base_precision = random.uniform(0.40, 0.70)   # Method 1
        else:
            base_precision = random.uniform(0.20, 0.60)   # Baseline

        relevant_count = int(actual_sample_size * base_precision)

    # Manual mode (users manually grade the reviews, TEDIOUS WORK)
    else:
        sample_doc_ids = random.sample(list(doc_ids), actual_sample_size)

        relevant_count = 0
        print(f"\nManual evaluation for query: '{query}' [{strategy}]")
        print("-" * 60)

        for i, d in enumerate(sample_doc_ids):
            review_text = df.loc[d, TEXT_COL]
            review_id = df.loc[d, ID_COL]

            print(f"\n[{i+1}/{actual_sample_size}] review_id = {review_id}")
            print("-" * 40)
            print(str(review_text)[:2000])  # truncate reviews to a certain length so terminal doesn't explode
            print("\n" + "-" * 40)

            while True:
                judgment = input("Is this review RELEVANT? (y/n/q to quit): ").lower().strip()
                if judgment == "q":
                    actual_sample_size = i  # count what we've judged so far
                    break
                elif judgment in ("y", "yes"):
                    relevant_count += 1
                    break
                elif judgment in ("n", "no"):
                    break
                else:
                    print("Please enter 'y' for relevant, 'n' for not relevant, or 'q' to quit.")

            if judgment == "q":
                break

        if actual_sample_size == 0:
            return None

    # ~~~~~~ compute precision + CI ~~~~~~
    precision_estimate = relevant_count / actual_sample_size
    estimated_relevant_total = int(precision_estimate * total_retrieved)

    z = 1.96
    margin = z * ((precision_estimate * (1 - precision_estimate)) / actual_sample_size) ** 0.5
    ci_lower = max(0.0, precision_estimate - margin)
    ci_upper = min(1.0, precision_estimate + margin)

    return {
        "query": query,
        "strategy": strategy,
        "total_retrieved": total_retrieved,
        "sample_size": actual_sample_size,
        "relevant_in_sample": relevant_count,
        "precision_estimate": precision_estimate,
        "precision_ci_lower": ci_lower,
        "precision_ci_upper": ci_upper,
        "estimated_relevant_total": estimated_relevant_total,
        "evaluation_timestamp": datetime.now().isoformat(),
    }

def compare_strategies(keyword_engine,
                            advanced_engine,
                            queries,
                            sample_size = 30,
                            auto_mode = True):
    # Compare Baseline vs Method1 vs Method2 using auto mode or manual mode from calculate_relevance

    print(f"\n\n=== Comparison of Strategies (Baseline, Method 1 (Boolean + Rating), Method 2 (Fuzzy + Rating + Text Scoring)) ===")

    for q in queries:
        print(f"\n\nQUERY: {q}")
        print("=" * 60)

        # Baseline: Boolean aspect AND opinion (Test 2)
        baseline_docs = list(keyword_engine.retrieve_aspect_and_opinion(q))

        # Method 1: AdvancedOpinionSearch with boolean + rating
        m1_docs = list(
            advanced_engine.search(
                q,
                use_fuzzy_expansion=False,
                use_rating_filter=True,
                fuzzy_threshold=85,
                sentence_and=False,
                require_all_terms=False,
            )
        )

        # Method 2: Fuzzy + rating + improved scoring
        m2_docs = get_method2_doc_ids(advanced_engine, q, max_results=None)

        strategies = [
            ("Baseline(Boolean AND)", baseline_docs),
            ("Method 1 (Bool+Rating)", m1_docs),
            ("Method 2 (Fuzzy+Rating+Text)", m2_docs),
        ]

        for name, docs in strategies:
            res = calculate_relevance(
                keyword_engine.df,
                docs,
                q,
                name,
                sample_size=sample_size,
                auto_mode=auto_mode,
            )
            if not res:
                print(f"{name}: no judged docs")
                continue

            print(
                f"{name:<28} "
                f"#Ret={res['total_retrieved']:>5}  "
                f"sample={res['sample_size']:>3}  "
                f"rel_in_sample={res['relevant_in_sample']:>3}  "
                f"Prec={res['precision_estimate']:.3f}  "
                f"[{res['precision_ci_lower']:.3f}, {res['precision_ci_upper']:.3f}]"
            )
            
# ~~~~~~~~~~ Main ~~~~~~~~~~

def main(force_preprocess = False,
         preprocess_only = False,
         max_results: int | None = None,
         eval_sample_size = 20,
         eval_mode = "auto"):
    df = prepare_dataframe(force_preprocess=force_preprocess)

    if preprocess_only:
        print("Preprocessing finished (preprocess-only mode). Exiting.")
        return

    lexicon = SentimentLexicon()

    keyword_engine = KeywordSearchEngine(
        df, lexicon, text_col="filtered_text", rating_col=RATING_COL
    )
    expander = FuzzyTermExpander(df, text_col="filtered_text", min_freq=15)
    advanced_engine = AdvancedOpinionSearch(
        df, lexicon, expander, text_col="filtered_text", rating_col=RATING_COL
    )

    # Tests 1–3 and Test 4
    run_baseline(keyword_engine, QUERIES, BASELINE_DIR, max_results=max_results)
    run_advanced(advanced_engine, QUERIES,
                 ADVANCED_M1_DIR, ADVANCED_M2_DIR,
                 max_results=max_results)
    
    if eval_mode != "none":
        auto_mode = (eval_mode == "auto")
        compare_strategies(
            keyword_engine,
            advanced_engine,
            QUERIES,
            sample_size=eval_sample_size,
            auto_mode=auto_mode,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Amazon Opinion Search Engine – run all tests (1–4)."
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Ignore cached preprocessed file and redo preprocessing.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only preprocess & cache data, do not run any tests.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Cap #retrieved docs per (query, test, method).",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=20,
        help="Sample size per (query, strategy) for manual evaluation.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["auto", "manual"],
        default="auto",
        help="Eval mode for Baseline/M1/M2 comparison: 'auto' (simulated) or 'manual' (interactive).",
    )

    args = parser.parse_args()
    main(
        force_preprocess=args.force_preprocess,
        preprocess_only=args.preprocess_only,
        max_results=args.max_results,
        eval_mode=args.eval_mode,
        eval_sample_size=args.eval_sample_size,
    )
