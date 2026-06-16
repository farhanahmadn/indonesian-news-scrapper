# 📰 End-to-End Indonesian News Intelligence Pipeline

A high-performance, asynchronous data engineering and AI pipeline designed to scrape, cluster, summarize, and predict the impact of news articles from Indonesia's Top Tier 1 News Portals in real-time. 

This system architecture consists of two main micro-workers: an **Asynchronous Data Scraper** and an **NLP & LLM Ingestion Pipeline**, both integrated natively with a Supabase PostgreSQL database.

---

## 🏗️ System Architecture & Workflow

1. **Data Acquisition (`scraper.py`)**: Fetches hourly news via concurrent RSS feed parsing, decodes Google News URLs, filters out junk content, and performs batch insertion into Supabase.
2. **Smart Clustering (`ai_pipeline.py` - Worker 3)**: Vectorizes new articles using a multilingual embedding model, matching them with active story clusters using Hybrid Semantic & Entity Similarity.
3. **GenAI Processing (`ai_pipeline.py` - Worker 4 & 5)**: Groups mature clusters, passes them through Groq API (Llama 3.1 8B), and extracts structured JSON containing journalistic summaries, actor sentiments, and sectoral risk predictions.

---

## 🚀 Tech Stack

- **Core Language:** Python 3.11+
- **Concurrency & Async I/O:** `asyncio`, `aiohttp`
- **Data Extraction:** `feedparser` (RSS), `trafilatura` (Web scraping & boilerplate removal)
- **AI & Natural Language Processing:**
  - `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`)
  - `scikit-learn` (Cosine Similarity metrics)
  - `groq` (Llama-3.1-8b-instant LLM orchestration)
  - `nltk` (Tokenization)
- **Database & Cloud Storage:** `supabase-py` (PostgreSQL Client with Batch Upsert operations)

---

## 💡 Key Features & Implementation Details

### 1. Asynchronous Scraper (`scraper.py`)
- **Non-blocking Operations:** Utilizing `asyncio.Semaphore(15)` to scrape hundreds of full-text articles concurrently without hitting rate limits or blocking server I/O bounds.
- **Smart Garbage Filter:** Pre-extraction regex heuristics to filter out tabular noise, weather forecasts, zodiacs, promotional coupon codes, and indexing pagination links.
- **Anti-Duplication:** Uses Supabase's database-level constraints with `upsert` to gracefully ignore duplicate URLs on conflict.

### 2. Lock-Centroid Smart Clustering (`ai_pipeline.py`)
- **Weighted Vectorization:** Combines article headings and lead paragraphs (70% Title weight / 30% Lead paragraph weight) to generate robust semantic embeddings.
- **Hybrid Similarity:** Employs a dual-pass constraint threshold. If semantic similarity scores fall between $0.45$ and $0.70$, the pipeline falls back to **Jaccard Similarity** on capitalized proper nouns (entities) to ensure precise matching.

### 3. Structured LLM Ingestion & Sentiment Analysis
- **Structured JSON Mode:** Forces the Llama 3.1 model to respond with guaranteed JSON objects for seamless database integration.
- **Relational Mapping:** Automatically maps abstract intensity levels ("Tinggi", "Sedang", "Rendah") into quantifiable percentages and normalized relational records for targeted sectors (e.g., Economy, Politics, Security).

---

## 📁 Repository Structure

```text
indonesian-news-intelligence/
│
├── .gitignore                  # Prevents env files from leaking
├── .env.example                # Configuration template for credentials
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── scraper.py                  # Worker 1: Async scraping & ingestion
└── ai_pipeline.py              # Worker 3, 4, 5: Clustering, Summarizing, ML-Inference
```

## 🛠️ Installation & Local Setup

1. **Clone the Repository**
    ```bash
    git clone [https://github.com/farhanahmadn/indonesian-news-scrapper.git](https://github.com/farhanahmadn/indonesian-news-scrapper.git)
    cd indonesian-news-scrapper
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Environment Configuration**
    Create a .env file in the root directory based on the .env.example template:

    ```Code snippet
    SUPABASE_URL=[https://your-project-id.supabase.co](https://your-project-id.supabase.co)
    SUPABASE_KEY=your-supabase-anon-or-service-role-key
    GROQ_API_KEY=your-groq-api-key-here
    ```

4. **Running the Pipeline**
    You can run both workers independently in separate terminal sessions or deploy them as background daemons:
    
    Run the News Aggregator (Runs periodically every 30 minutes):
    ```bash
    python scraper.py
    Run the AI & Enrichment Pipeline (Runs periodically every 30 minutes):
    ```
    
    ```bash
    python ai_pipeline.py
    ```

🔒 Security Note
> *This project adheres to professional security practices. All API credentials, access tokens, and cloud database strings are strictly managed via environment variables (python-dotenv) and ignored by Git tracking rules to avoid credential leaks* <.
