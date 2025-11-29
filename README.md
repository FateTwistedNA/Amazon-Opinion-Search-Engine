# Opinion Search Engine - Setup and Run Instructions

## Setup Instructions

### 1. Create Virtual Environment
```powershell
python -m venv .venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell / CMD):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```
source .venv/bin/activate
```

### 3. Install Required Libraries
```powershell
pip install pandas nltk python-levenshtein fuzzywuzzy
```
***Note***: If you run into errors with `fuzzywuzzy`, install `python-levenshtein` first and then reinstall `fuzzywuzzy`.
### 4. Download NLTK Data
Run this once to download required NLTK resources:
```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Database Configuration

### 5. Data Setup
Place the review dataset under the `data/` folder:

- `data/reviews_segment.pkl` – raw review data (provided by instructor)

- `data/reviews_segment_processed.pkl` – cached preprocessed version (auto-created on first run)

You don’t need to edit these paths, as `main.py` expects them exactly as above.

### 5.1 Repository Layout
```text
ProjectRoot/
├─ Codes/
│  ├─ main.py
│  ├─ keyword_search_engine.py
│  ├─ advanced_opinion_search.py
│  ├─ fuzzy_term_expander.py
│  ├─ sentiment_lexicon.py
│  ├─ results_export.py
│  └─ ...
├─ data/
│  ├─ reviews_segment.pkl
│  └─ reviews_segment_processed.pkl   # created automatically
└─ Outputs/
   ├─ Baseline/
   └─ AdvancedModel/
      ├─ Method1/
      └─ Method2/
```
Then cd to `Codes/` if you haven't:
```powershell
cd Codes
```

## Run the Program

### 6. Execute the Search Engine
```powershell
python main.py
```


The program will:
1. Loads (or builds) the preprocessed dataset
2. Runs Tests 1–3 (baseline Boolean)
3. Runs Test 4 for Method 1 and Method 2
4. Runs the strategy comparison (precision estimates)

Make sure your database is running and accessible before executing the program.

### 6.1 Rebuild preprocessing from scratch

If you want to ignore the cached `reviews_segment_processed.pkl` and redo preprocessing:
```powershell
python main.py --force-preprocess
```
If you only want to (re)build the preprocessed file and not run the tests:
```powershell
python main.py --force-preprocess --preprocess-only
```
