# Movie Feelings — Data Science Project

> **Decodelabs Data Science Internship** · All 5 Tasks Completed

A full data science pipeline applied to the **Movie Feelings Dataset** — 1,500 films spanning 1920–2024, each annotated with 50 emotion scores derived from plot summaries and GPT labelling.

---

## Project Structure

```
movie-feelings-ds/
│
├── movie_feelings_project.py      # Main script — all 5 tasks
├── movie_feelings_dataset.csv     # Primary dataset (1500 movies × 161 columns)
├── synonyms_of_feelings.csv       # Emotion synonym reference (50 emotions)
│
├── outputs/
│   ├── task4_visualizations.png   # 8-panel EDA visualization
│   └── task5_feature_importance.png  # Random Forest feature importances
│
└── README.md
```
---

## Dataset Description

| Column Group | Count | Description |
|---|---|---|
| Metadata | 9 | `imdb_id`, `title`, `year`, `plot`, `imdb_rating`, `tomatometer`, `metascore`, `avg_rating` |
| `f1_*` emotion scores | 50 | Continuous probability scores per emotion derived from plot text |
| `f2_*` emotion flags | 50 | Binary flags from plot-based labelling |
| `f3_*` emotion flags | 50 | Binary flags from GPT labelling |
| Label columns | 2 | `plot_feelings`, `gpt_feelings` (comma-separated emotion strings) |

**50 emotions tracked:** skepticism, serenity, fear, disgust, solidarity, elation, envy, puzzlement, recklessness, loyalty, trust, sadness, shame, catharsis, unease, introspection, excitement, riskiness, mischief, enlightenment, love, tension, sarcasm, frustration, longing, awe, defiance, amusement, vulnerability, surprise, compassion, resentment, triumph, happiness, acceptance, bravery, hope, conflict, exhilaration, inspiration, curiosity, regret, nostalgia, irony, despair, resignation, closeness, adrenaline, belonging, anger

---

## Task 1 — Data Collection & Dataset Understanding

**Goal:** Load the dataset and understand its full structure.

**Key findings:**
- **1,500 movies** spanning **1920–2024**
- **161 columns** across 5 logical groups
- Ratings: IMDB (44–93), Tomatometer (9–100), Metascore (16–100)
- Synonym file provides 18 rows of synonym variations for each of the 50 emotions

```python
df = pd.read_csv("movie_feelings_dataset.csv")
print(df.shape)        # (1500, 161)
print(df.dtypes)
print(df.describe())
```

---

## Task 2 — Data Cleaning & Preprocessing

**Goal:** Prepare the dataset for analysis.

**Steps taken:**

| Issue | Treatment |
|---|---|
| Missing `tomatometer` (26) & `metascore` (169) | Filled with column **median** |
| Missing `f1_*` emotion scores (27 each) | Filled with **0** (no signal = no probability) |
| Missing `plot` rows | **Dropped** (can't analyse feelings without text) |
| Duplicate rows | **None found** |
| `title` whitespace | Stripped with `.str.strip()` |
| `plot_feelings` / `gpt_feelings` text labels | Parsed into **Python lists** |

```python
# Example: filling emotion score NaNs
df[f1_cols] = df[f1_cols].fillna(0)

# Parsing label strings into lists
df["plot_feelings_list"] = df["plot_feelings"].str.split(r",\s*")
```

---

## Task 3 — Exploratory Data Analysis

**Goal:** Discover patterns and trends in the data.

### Rating Statistics (cleaned)

| Metric | IMDB | Tomatometer | Metascore | Avg Rating |
|---|---|---|---|---|
| Mean | 74.15 | 85.58 | 75.02 | 77.53 |
| Std | 6.24 | 13.52 | 14.26 | 8.86 |
| Min | 44 | 9 | 16 | 29.5 |
| Max | 93 | 100 | 100 | 95.25 |

### Top 10 Emotions (f1 average scores)

| Emotion | Avg Score |
|---|---|
|  Anger | 0.2462 |
|  Amusement | 0.1408 |
|  Awe | 0.1181 |
|  Surprise | 0.0967 |
|  Tension | 0.0936 |
|  Trust | 0.0914 |
|  Fear | 0.0857 |
|  Solidarity | 0.0782 |
|  Hope | 0.0711 |
|  Excitement | 0.0703 |

### Key Insights
- **Anger** is the single most dominant emotion across all movie plots
- **Rating sources correlate strongly** — Tomatometer & Metascore: r = 0.80
- **High-rated movies** (avg ≥ 80) share the same dominant emotion as the full dataset: anger
- **Movie production peaks** in the 2000s (262 films) and 1990s (246 films)
- **GPT top label** is `fear` (308 occurrences), followed by `unease` and `longing`

---

## Task 4 — Data Visualization

**8 visualizations generated** in a single figure (`task4_visualizations.png`):

| Panel | Chart Type | Shows |
|---|---|---|
| 1 | Histogram | Rating distributions (IMDB, Tomatometer, Metascore) |
| 2 | Horizontal bar | Top 10 average emotions (f1 scores) |
| 3 | Bar chart | Top 10 GPT-labelled emotions |
| 4 | Heatmap | Rating correlation matrix |
| 5 | Bar chart | Movie count per decade |
| 6 | Scatter plot | IMDB vs Tomatometer (coloured by avg rating) |
| 7 | Heatmap | Emotion intensity across decades |
| 8 | Pie chart | Top 6 GPT emotion share |

---

## Task 5 — Predictive Model

**Goal:** Predict a film's **dominant emotion** from its ratings and decade.

### Approach
- **Model:** Random Forest Classifier (150 trees)
- **Features:** `imdb_rating`, `tomatometer`, `metascore` + decade (one-hot encoded)
- **Target:** Dominant f1 emotion per movie (argmax of emotion scores)
- **Split:** 80% train / 20% test, stratified

### Results

| Metric | Value |
|---|---|
| **Accuracy** | **54.2%** |
| Classes | anger, amusement, awe, elation, excitement, fear, skepticism, tension |
| Best predicted | `anger` (precision 0.61, recall 0.84) |

> **Note:** 54% accuracy is a meaningful result for an 8-class emotion prediction task — especially given how subjective emotional labelling is. The model correctly identifies `anger` as the dominant class and partially resolves `amusement`.



## How to Run

```bash
# Clone or download the repo, then:
python movie_feelings_project.py
```

---

*Dataset: Movie Feelings Dataset · 1,500 films · 50 emotions · 3 annotation sources*
