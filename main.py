import os
import time
import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(title="Sentiment Analysis API")

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    pipeline = None

# --- Basic logging setup -----------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("ml_api")


def log_prediction_event(request_body: dict, prediction: dict, latency_ms: float):
    """Log a single prediction event as structured JSON.

    In a real production system, this would go to a log aggregator
    (e.g., CloudWatch, Datadog, ELK) rather than stdout — but the
    structured format is the same idea.
    """
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": request_body,
        "output": prediction,
        "latency_ms": round(latency_ms, 2),
    }
    logger.info(json.dumps(event))


def add_monitoring_middleware(app: FastAPI):
    """Attach a middleware that times every request and logs basic metrics."""

    @app.middleware("http")
    async def monitor_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
        }))

        return response

    return app


add_monitoring_middleware(app)


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to analyze")


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float | None = None
    probabilities: dict[str, float] | None = None


class BatchSentimentRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of review texts to analyze")


class BatchSentimentResponse(BaseModel):
    results: list[SentimentResponse]
    total: int
    positive_count: int
    negative_count: int


@app.get("/")
def root():
    return {
        "message": "Sentiment API is running. See /docs for usage, or /test for the interactive UI.",
        "ui_url": "/test",
        "docs_url": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    start_time = time.perf_counter()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    pred = pipeline.predict([request.text])[0]
    label = "positive" if pred == 1 else "negative"

    confidence = None
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([request.text])[0]
        confidence = round(float(probs.max()), 4)
        probabilities = {
            "negative": round(float(probs[0]), 4),
            "positive": round(float(probs[1]), 4),
        }

    latency_ms = (time.perf_counter() - start_time) * 1000
    res = SentimentResponse(sentiment=label, confidence=confidence, probabilities=probabilities)
    log_prediction_event(request.model_dump(), res.model_dump(), latency_ms)
    return res


@app.post("/predict/batch", response_model=BatchSentimentResponse)
def predict_batch(request: BatchSentimentRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not request.texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty.")

    preds = pipeline.predict(request.texts)
    has_proba = hasattr(pipeline, "predict_proba")
    probs = pipeline.predict_proba(request.texts) if has_proba else None

    results = []
    pos_count = 0
    neg_count = 0

    for i, pred in enumerate(preds):
        label = "positive" if pred == 1 else "negative"
        if label == "positive":
            pos_count += 1
        else:
            neg_count += 1

        conf = None
        prob_dict = None
        if has_proba:
            conf = round(float(probs[i].max()), 4)
            prob_dict = {
                "negative": round(float(probs[i][0]), 4),
                "positive": round(float(probs[i][1]), 4),
            }

        results.append(SentimentResponse(sentiment=label, confidence=conf, probabilities=prob_dict))

    return BatchSentimentResponse(
        results=results,
        total=len(results),
        positive_count=pos_count,
        negative_count=neg_count,
    )


# ---------------------------------------------------------------------------
# Frontend User Interface
# ---------------------------------------------------------------------------
HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SentimentScope • Production AI Sentiment Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --bg-surface: #111827;
      --bg-card: #141e33;
      --bg-input: #0c1322;
      --bg-hover: #1b2844;
      --border: rgba(255, 255, 255, 0.08);
      --border-hover: rgba(255, 255, 255, 0.16);
      --border-focus: #6366f1;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-sub: #64748b;
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --primary-glow: rgba(99, 102, 241, 0.25);
      --pos-color: #10b981;
      --pos-bg: rgba(16, 185, 129, 0.12);
      --pos-border: rgba(16, 185, 129, 0.35);
      --pos-glow: rgba(16, 185, 129, 0.2);
      --neg-color: #f43f5e;
      --neg-bg: rgba(244, 63, 94, 0.12);
      --neg-border: rgba(244, 63, 94, 0.35);
      --neg-glow: rgba(244, 63, 94, 0.2);
      --amber-color: #f59e0b;
      --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.25);
      --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.35);
      --shadow-glow: 0 0 30px rgba(99, 102, 241, 0.15);
      --radius-sm: 8px;
      --radius-md: 12px;
      --radius-lg: 18px;
      --radius-full: 9999px;
      --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
      --transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }

    [data-theme="light"] {
      --bg: #f8fafc;
      --bg-surface: #ffffff;
      --bg-card: #ffffff;
      --bg-input: #f8fafc;
      --bg-hover: #f1f5f9;
      --border: #e2e8f0;
      --border-hover: #cbd5e1;
      --border-focus: #6366f1;
      --text-main: #0f172a;
      --text-muted: #475569;
      --text-sub: #94a3b8;
      --primary: #4f46e5;
      --primary-hover: #4338ca;
      --primary-glow: rgba(79, 70, 229, 0.15);
      --pos-color: #059669;
      --pos-bg: rgba(5, 150, 105, 0.08);
      --pos-border: rgba(5, 150, 105, 0.25);
      --pos-glow: rgba(5, 150, 105, 0.15);
      --neg-color: #e11d48;
      --neg-bg: rgba(225, 29, 72, 0.08);
      --neg-border: rgba(225, 29, 72, 0.25);
      --neg-glow: rgba(225, 29, 72, 0.15);
      --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.04);
      --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.06);
      --shadow-glow: 0 0 30px rgba(79, 70, 229, 0.08);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: var(--font-sans);
      background-color: var(--bg);
      color: var(--text-main);
      min-height: 100vh;
      line-height: 1.5;
      transition: background-color 0.3s ease, color 0.3s ease;
      background-image: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 60%);
      background-repeat: no-repeat;
    }

    /* Top Navigation Bar */
    header.navbar {
      border-bottom: 1px solid var(--border);
      background-color: rgba(17, 24, 39, 0.7);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 0.75rem 1.5rem;
      transition: background-color 0.3s ease;
    }

    [data-theme="light"] header.navbar {
      background-color: rgba(255, 255, 255, 0.85);
    }

    .nav-container {
      max-width: 1180px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      text-decoration: none;
      color: inherit;
    }

    .brand-icon {
      width: 36px;
      height: 36px;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    }

    .brand-info h1 {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      line-height: 1.1;
    }

    .brand-info span {
      font-size: 0.75rem;
      color: var(--text-muted);
      font-weight: 500;
    }

    .nav-actions {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      padding: 0.3rem 0.75rem;
      border-radius: var(--radius-full);
      font-size: 0.78rem;
      font-weight: 600;
      background: var(--bg-hover);
      border: 1px solid var(--border);
      color: var(--text-muted);
      transition: var(--transition);
    }

    .status-badge.online {
      color: var(--pos-color);
      border-color: var(--pos-border);
      background: var(--pos-bg);
    }

    .status-badge.offline {
      color: var(--neg-color);
      border-color: var(--neg-border);
      background: var(--neg-bg);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
      box-shadow: 0 0 8px currentColor;
    }

    .nav-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      color: var(--text-main);
      padding: 0.45rem 0.8rem;
      border-radius: var(--radius-md);
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      transition: var(--transition);
    }

    .nav-btn:hover {
      background: var(--bg-hover);
      border-color: var(--border-hover);
      transform: translateY(-1px);
    }

    /* Main Container */
    main.main-content {
      max-width: 1180px;
      margin: 0 auto;
      padding: 2.25rem 1.5rem 4rem 1.5rem;
    }

    /* Hero Section */
    .hero-banner {
      margin-bottom: 2rem;
      text-align: center;
    }

    .hero-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.25rem 0.75rem;
      border-radius: var(--radius-full);
      background: rgba(99, 102, 241, 0.1);
      border: 1px solid rgba(99, 102, 241, 0.25);
      color: #818cf8;
      font-size: 0.8rem;
      font-weight: 600;
      margin-bottom: 0.85rem;
    }

    .hero-banner h2 {
      font-size: 2.35rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      line-height: 1.2;
      margin-bottom: 0.6rem;
      background: linear-gradient(135deg, var(--text-main) 40%, var(--text-muted) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .hero-banner p {
      font-size: 1.02rem;
      color: var(--text-muted);
      max-width: 640px;
      margin: 0 auto;
    }

    /* Tabs Navigation */
    .tab-bar {
      display: flex;
      gap: 0.5rem;
      background: var(--bg-surface);
      padding: 0.35rem;
      border-radius: var(--radius-lg);
      border: 1px solid var(--border);
      max-width: 480px;
      margin: 0 auto 2.25rem auto;
    }

    .tab-btn {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      padding: 0.6rem 1rem;
      background: transparent;
      border: none;
      border-radius: var(--radius-md);
      color: var(--text-muted);
      font-family: inherit;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      transition: var(--transition);
    }

    .tab-btn:hover {
      color: var(--text-main);
    }

    .tab-btn.active {
      background: var(--primary);
      color: #ffffff;
      box-shadow: 0 2px 10px var(--primary-glow);
    }

    /* Tab Panes */
    .tab-content {
      display: none;
      animation: fadeIn 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .tab-content.active {
      display: block;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Layout Grid for Single Analysis */
    .studio-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.75rem;
      align-items: start;
    }

    @media (max-width: 900px) {
      .studio-grid {
        grid-template-columns: 1fr;
      }
    }

    /* Cards */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 1.5rem;
      box-shadow: var(--shadow-sm);
      transition: var(--transition);
      position: relative;
    }

    .card:hover {
      border-color: var(--border-hover);
    }

    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.25rem;
    }

    .card-title {
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .card-title svg {
      color: var(--primary);
    }

    /* Quick Example Chips */
    .presets-label {
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-sub);
      margin-bottom: 0.6rem;
    }

    .preset-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 1.25rem;
    }

    .chip-btn {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: var(--radius-full);
      padding: 0.35rem 0.75rem;
      font-size: 0.78rem;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      transition: var(--transition);
      font-family: inherit;
    }

    .chip-btn:hover {
      background: var(--bg-hover);
      color: var(--text-main);
      border-color: var(--border-hover);
      transform: translateY(-1px);
    }

    /* Textarea & Inputs */
    .textarea-wrapper {
      position: relative;
      margin-bottom: 1rem;
    }

    textarea.sentiment-input {
      width: 100%;
      min-height: 150px;
      padding: 1rem;
      font-family: var(--font-sans);
      font-size: 0.95rem;
      line-height: 1.55;
      background: var(--bg-input);
      border: 1.5px solid var(--border);
      border-radius: var(--radius-md);
      color: var(--text-main);
      resize: vertical;
      transition: var(--transition);
      outline: none;
    }

    textarea.sentiment-input:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px var(--primary-glow);
    }

    .input-meta-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.78rem;
      color: var(--text-sub);
      margin-top: 0.4rem;
      padding: 0 0.25rem;
    }

    .input-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      margin-top: 1.25rem;
      flex-wrap: wrap;
    }

    .left-actions {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .live-toggle {
      display: flex;
      align-items: center;
      gap: 0.45rem;
      font-size: 0.82rem;
      color: var(--text-muted);
      cursor: pointer;
      user-select: none;
    }

    .live-toggle input {
      accent-color: var(--primary);
      cursor: pointer;
    }

    .btn-primary {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      color: #ffffff;
      border: none;
      border-radius: var(--radius-md);
      padding: 0.65rem 1.35rem;
      font-size: 0.92rem;
      font-weight: 600;
      font-family: inherit;
      cursor: pointer;
      box-shadow: 0 4px 14px var(--primary-glow);
      transition: var(--transition);
    }

    .btn-primary:hover {
      background: linear-gradient(135deg, #4f46e5, #4338ca);
      transform: translateY(-1.5px);
      box-shadow: 0 6px 20px var(--primary-glow);
    }

    .btn-primary:active {
      transform: translateY(0);
    }

    .btn-secondary {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: var(--radius-md);
      padding: 0.6rem 1rem;
      font-size: 0.88rem;
      font-weight: 600;
      font-family: inherit;
      cursor: pointer;
      transition: var(--transition);
    }

    .btn-secondary:hover {
      background: var(--bg-hover);
      color: var(--text-main);
      border-color: var(--border-hover);
    }

    /* Result State Styling */
    .empty-result-state {
      padding: 3.5rem 1.5rem;
      text-align: center;
      color: var(--text-sub);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 0.75rem;
      border: 1.5px dashed var(--border);
      border-radius: var(--radius-md);
    }

    .empty-result-state svg {
      opacity: 0.4;
    }

    .verdict-card {
      padding: 1.25rem;
      border-radius: var(--radius-md);
      display: flex;
      align-items: center;
      gap: 1.25rem;
      margin-bottom: 1.5rem;
      transition: var(--transition);
      border: 1px solid transparent;
    }

    .verdict-card.positive {
      background: var(--pos-bg);
      border-color: var(--pos-border);
      box-shadow: 0 0 24px var(--pos-glow);
    }

    .verdict-card.negative {
      background: var(--neg-bg);
      border-color: var(--neg-border);
      box-shadow: 0 0 24px var(--neg-glow);
    }

    .verdict-icon {
      font-size: 2.8rem;
      line-height: 1;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .verdict-details h3 {
      font-size: 1.45rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      text-transform: uppercase;
      line-height: 1.15;
    }

    .verdict-card.positive .verdict-details h3 {
      color: var(--pos-color);
    }

    .verdict-card.negative .verdict-details h3 {
      color: var(--neg-color);
    }

    .verdict-details p {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 0.2rem;
      font-weight: 500;
    }

    /* Confidence Bar */
    .confidence-section {
      margin-bottom: 1.5rem;
    }

    .meter-label-row {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 0.5rem;
    }

    .meter-title {
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--text-muted);
    }

    .meter-value {
      font-size: 1.35rem;
      font-weight: 800;
      font-family: var(--font-mono);
      letter-spacing: -0.02em;
    }

    .confidence-bar-track {
      height: 14px;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-full);
      overflow: hidden;
      display: flex;
      position: relative;
    }

    .confidence-bar-fill-pos {
      height: 100%;
      background: linear-gradient(90deg, #10b981, #34d399);
      transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .confidence-bar-fill-neg {
      height: 100%;
      background: linear-gradient(90deg, #f43f5e, #fb7185);
      transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .bar-legend {
      display: flex;
      justify-content: space-between;
      font-size: 0.78rem;
      font-weight: 600;
      margin-top: 0.45rem;
    }

    .legend-pos {
      color: var(--pos-color);
    }

    .legend-neg {
      color: var(--neg-color);
    }

    /* Metrics Grid */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.75rem;
      margin-bottom: 1.5rem;
    }

    .metric-box {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 0.85rem;
      text-align: center;
    }

    .metric-box-title {
      font-size: 0.72rem;
      color: var(--text-sub);
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.04em;
      margin-bottom: 0.25rem;
    }

    .metric-box-value {
      font-size: 1.05rem;
      font-weight: 700;
      font-family: var(--font-mono);
      color: var(--text-main);
    }

    /* History Session Card */
    .history-card {
      margin-top: 2rem;
    }

    .history-table-container {
      overflow-x: auto;
      max-height: 280px;
    }

    table.styled-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      text-align: left;
    }

    table.styled-table th {
      padding: 0.75rem 1rem;
      background: var(--bg-surface);
      color: var(--text-sub);
      font-weight: 600;
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 2;
    }

    table.styled-table td {
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
      color: var(--text-main);
    }

    table.styled-table tr:hover td {
      background: var(--bg-hover);
    }

    .tag-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0.2rem 0.6rem;
      border-radius: var(--radius-full);
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
    }

    .tag-badge.positive {
      background: var(--pos-bg);
      color: var(--pos-color);
      border: 1px solid var(--pos-border);
    }

    .tag-badge.negative {
      background: var(--neg-bg);
      color: var(--neg-color);
      border: 1px solid var(--neg-border);
    }

    /* Batch Processing */
    .batch-summary-cards {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    @media (max-width: 800px) {
      .batch-summary-cards {
        grid-template-columns: repeat(2, 1fr);
      }
    }

    .summary-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 1rem;
    }

    .summary-card-label {
      font-size: 0.75rem;
      color: var(--text-sub);
      text-transform: uppercase;
      font-weight: 600;
    }

    .summary-card-number {
      font-size: 1.6rem;
      font-weight: 800;
      font-family: var(--font-mono);
      margin-top: 0.25rem;
    }

    /* API Inspector Styles */
    .code-block {
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 1rem;
      font-family: var(--font-mono);
      font-size: 0.85rem;
      color: #38bdf8;
      overflow-x: auto;
      position: relative;
      margin-bottom: 1.25rem;
    }

    .code-copy-btn {
      position: absolute;
      top: 0.75rem;
      right: 0.75rem;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: var(--radius-sm);
      padding: 0.35rem 0.65rem;
      font-size: 0.75rem;
      cursor: pointer;
      transition: var(--transition);
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
    }

    .code-copy-btn:hover {
      background: var(--bg-hover);
      color: var(--text-main);
    }

    /* Toast notification */
    #toast-container {
      position: fixed;
      bottom: 1.5rem;
      right: 1.5rem;
      z-index: 1000;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .toast {
      background: var(--bg-card);
      border: 1px solid var(--border-hover);
      box-shadow: var(--shadow-md);
      color: var(--text-main);
      padding: 0.75rem 1.25rem;
      border-radius: var(--radius-md);
      font-size: 0.88rem;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 0.6rem;
      animation: slideInRight 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }

    @keyframes slideInRight {
      from { transform: translateX(100%); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }

    /* Spinner */
    .spinner {
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: #ffffff;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    /* Footer */
    footer.app-footer {
      border-top: 1px solid var(--border);
      padding: 2rem 1.5rem;
      margin-top: 4rem;
      text-align: center;
      color: var(--text-sub);
      font-size: 0.85rem;
    }

    footer.app-footer a {
      color: var(--primary);
      text-decoration: none;
    }

    footer.app-footer a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>

  <!-- Navigation Bar -->
  <header class="navbar">
    <div class="nav-container">
      <a href="/test" class="brand">
        <div class="brand-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
          </svg>
        </div>
        <div class="brand-info">
          <h1>SentimentScope</h1>
          <span>FastAPI • Scikit-Learn NLP Production Service</span>
        </div>
      </a>

      <div class="nav-actions">
        <div id="healthBadge" class="status-badge" title="Model inference readiness">
          <span class="status-dot"></span>
          <span id="healthText">Connecting...</span>
        </div>
        <a href="/docs" target="_blank" class="nav-btn" title="Open Swagger OpenAPI Documentation">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
          </svg>
          API Docs
        </a>
        <button id="themeToggleBtn" class="nav-btn" title="Toggle Dark/Light Mode" onclick="toggleTheme()">
          <span id="themeIcon">☀️</span>
        </button>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="main-content">

    <!-- Hero Banner -->
    <div class="hero-banner">
      <div class="hero-pill">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
        </svg>
        Live Production Inference
      </div>
      <h2>Sentiment Intelligence Studio</h2>
      <p>Analyze emotion and polarity in product reviews, comments, and feedback in real time with high-speed TF-IDF + Logistic Regression classification.</p>
    </div>

    <!-- Navigation Tab Bar -->
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('single')">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/><path d="M8 12h8"/><path d="M12 8v8"/>
        </svg>
        Single Review
      </button>
      <button class="tab-btn" onclick="switchTab('batch')">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
        </svg>
        Batch Testing
      </button>
      <button class="tab-btn" onclick="switchTab('api')">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>
        </svg>
        cURL & API
      </button>
    </div>

    <!-- Tab 1: Single Review Analysis -->
    <div id="tab-single" class="tab-content active">
      <div class="studio-grid">

        <!-- Input Card -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
              Input Review
            </span>
            <button class="btn-secondary" style="padding: 0.35rem 0.7rem; font-size: 0.78rem;" onclick="clearInput()">
              ✕ Clear
            </button>
          </div>

          <!-- Presets -->
          <div class="presets-label">Quick Sample Presets</div>
          <div class="preset-chips">
            <button class="chip-btn" onclick="loadSample('This product completely exceeded my expectations! Brilliant build quality and fast shipping.')">
              🌟 Outstanding
            </button>
            <button class="chip-btn" onclick="loadSample('Terrible customer service and the item arrived broken. Very disappointed with this purchase.')">
              💔 Broken & Awful
            </button>
            <button class="chip-btn" onclick="loadSample('The battery life is decent, but the screen quality could definitely be improved.')">
              ⚖️ Mixed Review
            </button>
            <button class="chip-btn" onclick="loadSample('An absolute cinematic masterpiece with breathtaking visual storytelling and memorable acting!')">
              🎬 Masterpiece
            </button>
            <button class="chip-btn" onclick="loadSample('Horribly delayed delivery, defective parts, and support refused to issue a refund. Avoid!')">
              ⛔ Total Disaster
            </button>
          </div>

          <!-- Textarea -->
          <div class="textarea-wrapper">
            <textarea id="reviewText" class="sentiment-input" placeholder="Type or paste any customer review, product feedback, or sentence to evaluate its sentiment..." oninput="onInputChanged()"></textarea>
            <div class="input-meta-bar">
              <div>
                <span id="charCount">0</span> characters &bull; <span id="wordCount">0</span> words
              </div>
              <div style="font-size: 0.74rem;">
                Shortcut: <kbd style="background:var(--bg-hover); padding:2px 4px; border-radius:4px;">⌘/Ctrl + Enter</kbd>
              </div>
            </div>
          </div>

          <!-- Action Bar -->
          <div class="input-actions">
            <div class="left-actions">
              <label class="live-toggle" title="Automatically evaluate as you type">
                <input type="checkbox" id="liveToggle" onchange="onLiveToggleChanged()">
                <span>Live inference</span>
              </label>
            </div>
            <button id="analyzeBtn" class="btn-primary" onclick="predictSingle()">
              <span id="btnIcon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                  <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
                </svg>
              </span>
              <span id="btnLabel">Analyze Sentiment</span>
            </button>
          </div>
        </div>

        <!-- Result Card -->
        <div class="card" id="resultCard">
          <div class="card-header">
            <span class="card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                <polyline points="22 4 12 14.01 9 11.01"/>
              </svg>
              Prediction Result
            </span>
            <span id="latencyBadge" class="status-badge" style="display:none;"></span>
          </div>

          <!-- Placeholder when no result -->
          <div id="emptyResultState" class="empty-result-state">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="8" x2="12" y2="12"/>
              <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            <div style="font-weight: 600; color: var(--text-main);">No Analysis Yet</div>
            <p style="font-size: 0.85rem; max-width: 320px;">
              Enter your text or select a preset sample on the left to see sentiment polarity and confidence scores.
            </p>
          </div>

          <!-- Active Result Content -->
          <div id="activeResultState" style="display: none;">

            <!-- Verdict Banner -->
            <div id="verdictCard" class="verdict-card positive">
              <div id="verdictIcon" class="verdict-icon">😊</div>
              <div class="verdict-details">
                <h3 id="verdictSentiment">POSITIVE</h3>
                <p id="verdictSubtitle">High Confidence Detection</p>
              </div>
            </div>

            <!-- Confidence Bar -->
            <div class="confidence-section">
              <div class="meter-label-row">
                <span class="meter-title">Classification Confidence</span>
                <span id="confidencePercentage" class="meter-value">94.8%</span>
              </div>
              <div class="confidence-bar-track">
                <div id="barPos" class="confidence-bar-fill-pos" style="width: 50%;" title="Positive probability"></div>
                <div id="barNeg" class="confidence-bar-fill-neg" style="width: 50%;" title="Negative probability"></div>
              </div>
              <div class="bar-legend">
                <span class="legend-pos" id="labelPos">Positive: 50.0%</span>
                <span class="legend-neg" id="labelNeg">Negative: 50.0%</span>
              </div>
            </div>

            <!-- Performance & Metric Tiles -->
            <div class="metrics-grid">
              <div class="metric-box">
                <div class="metric-box-title">Inference Time</div>
                <div id="metricLatency" class="metric-box-value">-- ms</div>
              </div>
              <div class="metric-box">
                <div class="metric-box-title">Word Count</div>
                <div id="metricWords" class="metric-box-value">0</div>
              </div>
              <div class="metric-box">
                <div class="metric-box-title">Raw Class</div>
                <div id="metricClass" class="metric-box-value">--</div>
              </div>
            </div>

            <div style="display: flex; gap: 0.5rem; justify-content: flex-end;">
              <button class="btn-secondary" style="font-size: 0.8rem; padding: 0.45rem 0.8rem;" onclick="copyResultSummary()">
                📋 Copy Result
              </button>
            </div>
          </div>

        </div>

      </div>

      <!-- History Card -->
      <div class="card history-card">
        <div class="card-header">
          <span class="card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
            </svg>
            Session History
          </span>
          <button class="btn-secondary" style="padding: 0.35rem 0.75rem; font-size: 0.78rem;" onclick="clearHistory()">
            Clear History
          </button>
        </div>
        <div class="history-table-container">
          <table class="styled-table" id="historyTable">
            <thead>
              <tr>
                <th style="width: 120px;">Sentiment</th>
                <th style="width: 110px;">Confidence</th>
                <th>Review Snippet</th>
                <th style="width: 90px; text-align: right;">Action</th>
              </tr>
            </thead>
            <tbody id="historyBody">
              <tr>
                <td colspan="4" style="text-align: center; color: var(--text-sub); padding: 1.5rem;">
                  No previous evaluations yet.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Tab 2: Batch Analysis -->
    <div id="tab-batch" class="tab-content">
      <div class="card" style="margin-bottom: 1.75rem;">
        <div class="card-header">
          <span class="card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
            </svg>
            Batch Review Input
          </span>
          <button class="btn-secondary" style="font-size: 0.8rem; padding: 0.35rem 0.75rem;" onclick="loadBatchSample()">
            Load 5 Sample Reviews
          </button>
        </div>
        <textarea id="batchInput" class="sentiment-input" style="min-height: 140px;" placeholder="Paste multiple lines of reviews here (one review sentence per line)..."></textarea>
        <div class="input-actions" style="margin-top: 1rem;">
          <span style="font-size: 0.82rem; color: var(--text-sub);">
            Lines: <span id="batchLineCount">0</span>
          </span>
          <button id="batchAnalyzeBtn" class="btn-primary" onclick="predictBatch()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
            Run Batch Analysis
          </button>
        </div>
      </div>

      <!-- Batch Results Section -->
      <div id="batchResultsSection" style="display: none;">
        <div class="batch-summary-cards">
          <div class="summary-card">
            <div class="summary-card-label">Total Analyzed</div>
            <div id="bTotal" class="summary-card-number">0</div>
          </div>
          <div class="summary-card">
            <div class="summary-card-label" style="color: var(--pos-color);">Positive Reviews</div>
            <div id="bPos" class="summary-card-number" style="color: var(--pos-color);">0</div>
          </div>
          <div class="summary-card">
            <div class="summary-card-label" style="color: var(--neg-color);">Negative Reviews</div>
            <div id="bNeg" class="summary-card-number" style="color: var(--neg-color);">0</div>
          </div>
          <div class="summary-card">
            <div class="summary-card-label">Positive Ratio</div>
            <div id="bRatio" class="summary-card-number">0%</div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-title">Batch Breakdown</span>
            <button class="btn-secondary" style="font-size: 0.8rem; padding: 0.4rem 0.8rem;" onclick="exportBatchCSV()">
              📥 Export CSV
            </button>
          </div>
          <div style="overflow-x: auto; max-height: 360px;">
            <table class="styled-table" id="batchTable">
              <thead>
                <tr>
                  <th style="width: 50px;">#</th>
                  <th style="width: 120px;">Verdict</th>
                  <th style="width: 110px;">Confidence</th>
                  <th>Review Text</th>
                </tr>
              </thead>
              <tbody id="batchTableBody"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- Tab 3: API & cURL Inspector -->
    <div id="tab-api" class="tab-content">
      <div class="card" style="margin-bottom: 1.5rem;">
        <div class="card-header">
          <span class="card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="4 17 10 11 4 5"/>
              <line x1="12" y1="19" x2="20" y2="19"/>
            </svg>
            Live cURL Command
          </span>
          <span class="status-badge online">POST /predict</span>
        </div>
        <p style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 0.75rem;">
          This cURL snippet is generated dynamically using your current single review input:
        </p>
        <div class="code-block">
          <button class="code-copy-btn" onclick="copyCode('curlCode')">Copy</button>
          <pre id="curlCode">curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product completely exceeded my expectations!"}'</pre>
        </div>
      </div>

      <div class="studio-grid">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Python Integration</span>
          </div>
          <div class="code-block">
            <button class="code-copy-btn" onclick="copyCode('pyCode')">Copy</button>
            <pre id="pyCode">import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Outstanding service and speed!"}
)
data = response.json()
print("Sentiment:", data["sentiment"])
print("Confidence:", data["confidence"])</pre>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-title">JavaScript Fetch</span>
          </div>
          <div class="code-block">
            <button class="code-copy-btn" onclick="copyCode('jsCode')">Copy</button>
            <pre id="jsCode">const res = await fetch("/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "Super fast and accurate!" })
});
const data = await res.json();
console.log(data);</pre>
          </div>
        </div>
      </div>
    </div>

  </main>

  <!-- Footer -->
  <footer class="app-footer">
    <p>
      Sentiment Analysis Production Service &bull; <a href="/docs" target="_blank">Swagger OpenAPI</a> &bull; <a href="/health" target="_blank">Health Check</a>
    </p>
    <p style="margin-top: 0.35rem; font-size: 0.78rem;">
      Trained with scikit-learn TF-IDF Vectorizer + Logistic Regression &bull; Powered by FastAPI
    </p>
  </footer>

  <!-- Toast Container -->
  <div id="toast-container"></div>

  <!-- JavaScript Application Logic -->
  <script>
    let currentTheme = localStorage.getItem("theme") || "dark";
    document.documentElement.setAttribute("data-theme", currentTheme);
    updateThemeIcon();

    let liveTypingTimer = null;
    let lastEvaluatedText = "";
    let historyItems = JSON.parse(localStorage.getItem("sentiment_history") || "[]");
    let lastBatchResults = [];

    // Initialize UI on DOM load
    window.addEventListener("DOMContentLoaded", () => {
      checkHealth();
      renderHistory();
      updateCurlPreview();

      // Auto-load first sample if empty
      const reviewText = document.getElementById("reviewText");
      if (!reviewText.value) {
        reviewText.value = "This product completely exceeded my expectations! Brilliant build quality and fast shipping.";
        onInputChanged();
      }

      // Shortcut: Cmd/Ctrl + Enter
      document.addEventListener("keydown", (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
          const activeTab = document.querySelector(".tab-content.active");
          if (activeTab && activeTab.id === "tab-single") {
            predictSingle();
          } else if (activeTab && activeTab.id === "tab-batch") {
            predictBatch();
          }
        }
      });
    });

    // Theme Toggle
    function toggleTheme() {
      currentTheme = currentTheme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", currentTheme);
      localStorage.setItem("theme", currentTheme);
      updateThemeIcon();
    }

    function updateThemeIcon() {
      const icon = document.getElementById("themeIcon");
      if (icon) {
        icon.textContent = currentTheme === "dark" ? "☀️" : "🌙";
      }
    }

    // Health check polling
    async function checkHealth() {
      const badge = document.getElementById("healthBadge");
      const text = document.getElementById("healthText");
      try {
        const res = await fetch("/health");
        const data = await res.json();
        if (data.status === "ok" && data.model_loaded) {
          badge.className = "status-badge online";
          text.textContent = "Model Ready";
        } else {
          badge.className = "status-badge offline";
          text.textContent = "Model Missing";
        }
      } catch (err) {
        badge.className = "status-badge offline";
        text.textContent = "Server Offline";
      }
    }

    // Switch Tabs
    function switchTab(tabName) {
      document.querySelectorAll(".tab-btn").forEach((btn, idx) => {
        const names = ["single", "batch", "api"];
        if (names[idx] === tabName) {
          btn.classList.add("active");
        } else {
          btn.classList.remove("active");
        }
      });

      document.querySelectorAll(".tab-content").forEach(pane => {
        pane.classList.remove("active");
      });
      const targetPane = document.getElementById("tab-" + tabName);
      if (targetPane) targetPane.classList.add("active");
    }

    // Toast notification
    function showToast(message, icon = "ℹ️") {
      const container = document.getElementById("toast-container");
      const toast = document.createElement("div");
      toast.className = "toast";
      toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
      container.appendChild(toast);
      setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(100%)";
        toast.style.transition = "all 0.3s ease";
        setTimeout(() => toast.remove(), 300);
      }, 2600);
    }

    // Copy to clipboard helper
    function copyCode(elementId) {
      const text = document.getElementById(elementId).innerText;
      navigator.clipboard.writeText(text).then(() => {
        showToast("Copied code to clipboard!", "📋");
      });
    }

    // Sample Loader
    function loadSample(text) {
      const ta = document.getElementById("reviewText");
      ta.value = text;
      onInputChanged();
      predictSingle();
      showToast("Preset sample loaded", "✨");
    }

    function clearInput() {
      const ta = document.getElementById("reviewText");
      ta.value = "";
      onInputChanged();
      document.getElementById("emptyResultState").style.display = "flex";
      document.getElementById("activeResultState").style.display = "none";
      document.getElementById("latencyBadge").style.display = "none";
      ta.focus();
    }

    // Textarea input watcher
    function onInputChanged() {
      const text = document.getElementById("reviewText").value;
      const chars = text.length;
      const words = text.trim() ? text.trim().split(/\s+/).length : 0;
      document.getElementById("charCount").textContent = chars;
      document.getElementById("wordCount").textContent = words;
      updateCurlPreview();

      const liveChecked = document.getElementById("liveToggle").checked;
      if (liveChecked && text.trim().length > 3 && text !== lastEvaluatedText) {
        clearTimeout(liveTypingTimer);
        liveTypingTimer = setTimeout(() => {
          predictSingle();
        }, 380);
      }
    }

    function onLiveToggleChanged() {
      const liveChecked = document.getElementById("liveToggle").checked;
      if (liveChecked) {
        showToast("Live prediction enabled", "⚡");
        predictSingle();
      }
    }

    function updateCurlPreview() {
      const text = document.getElementById("reviewText").value || "This product is fantastic!";
      const safeText = text.replace(/"/g, '\\"');
      const curlSnippet = `curl -X POST "http://localhost:8000/predict" \\\n  -H "Content-Type: application/json" \\\n  -d '{"text": "${safeText}"}'`;
      const el = document.getElementById("curlCode");
      if (el) el.innerText = curlSnippet;
    }

    // Single Prediction API Call
    async function predictSingle() {
      const ta = document.getElementById("reviewText");
      const text = ta.value.trim();
      if (!text) {
        showToast("Please enter some text to analyze", "⚠️");
        ta.focus();
        return;
      }

      const btn = document.getElementById("analyzeBtn");
      const btnIcon = document.getElementById("btnIcon");
      const btnLabel = document.getElementById("btnLabel");

      btn.disabled = true;
      btnIcon.innerHTML = `<span class="spinner"></span>`;
      btnLabel.textContent = "Analyzing...";

      const startTime = performance.now();

      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text })
        });

        const latency = Math.round(performance.now() - startTime);

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || "Server error: " + res.status);
        }

        const data = await res.json();
        lastEvaluatedText = text;
        renderResult(data, latency, text);
        addToHistory(data, text);

      } catch (err) {
        showToast(err.message, "❌");
      } finally {
        btn.disabled = false;
        btnIcon.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`;
        btnLabel.textContent = "Analyze Sentiment";
      }
    }

    // Render single prediction result
    function renderResult(data, latency, text) {
      document.getElementById("emptyResultState").style.display = "none";
      const activeState = document.getElementById("activeResultState");
      activeState.style.display = "block";

      const verdictCard = document.getElementById("verdictCard");
      const verdictIcon = document.getElementById("verdictIcon");
      const verdictSentiment = document.getElementById("verdictSentiment");
      const verdictSubtitle = document.getElementById("verdictSubtitle");

      const isPos = data.sentiment.toLowerCase() === "positive";
      verdictCard.className = "verdict-card " + (isPos ? "positive" : "negative");
      verdictIcon.textContent = isPos ? "😊" : "🙁";
      verdictSentiment.textContent = isPos ? "POSITIVE SENTIMENT" : "NEGATIVE SENTIMENT";

      const conf = data.confidence !== null && data.confidence !== undefined ? data.confidence : 1.0;
      const confPct = (conf * 100).toFixed(1) + "%";

      let confidenceDesc = "High Confidence Detection";
      if (conf < 0.65) confidenceDesc = "Low / Ambiguous Confidence";
      else if (conf < 0.85) confidenceDesc = "Moderate Confidence Detection";
      verdictSubtitle.textContent = confidenceDesc;

      document.getElementById("confidencePercentage").textContent = confPct;

      let posProb = 0.5;
      let negProb = 0.5;
      if (data.probabilities) {
        posProb = data.probabilities.positive;
        negProb = data.probabilities.negative;
      } else {
        posProb = isPos ? conf : 1 - conf;
        negProb = 1 - posProb;
      }

      const posWidth = Math.round(posProb * 100);
      const negWidth = 100 - posWidth;

      document.getElementById("barPos").style.width = posWidth + "%";
      document.getElementById("barNeg").style.width = negWidth + "%";
      document.getElementById("labelPos").textContent = "Positive: " + (posProb * 100).toFixed(1) + "%";
      document.getElementById("labelNeg").textContent = "Negative: " + (negProb * 100).toFixed(1) + "%";

      // Latency badge
      const latencyBadge = document.getElementById("latencyBadge");
      latencyBadge.style.display = "inline-flex";
      latencyBadge.className = "status-badge online";
      latencyBadge.innerHTML = `<span class="status-dot"></span>⚡ ${latency} ms`;

      // Metrics
      document.getElementById("metricLatency").textContent = latency + " ms";
      const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
      document.getElementById("metricWords").textContent = wordCount;
      document.getElementById("metricClass").textContent = isPos ? "Class 1" : "Class 0";
    }

    function copyResultSummary() {
      const sentiment = document.getElementById("verdictSentiment").textContent;
      const conf = document.getElementById("confidencePercentage").textContent;
      const text = document.getElementById("reviewText").value;
      const summary = `Sentiment: ${sentiment} (${conf})\nReview: "${text}"`;
      navigator.clipboard.writeText(summary).then(() => {
        showToast("Result summary copied to clipboard!", "📋");
      });
    }

    // History management
    function addToHistory(data, text) {
      const item = {
        sentiment: data.sentiment,
        confidence: data.confidence,
        text: text,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      historyItems.unshift(item);
      if (historyItems.length > 10) historyItems.pop();
      localStorage.setItem("sentiment_history", JSON.stringify(historyItems));
      renderHistory();
    }

    function renderHistory() {
      const tbody = document.getElementById("historyBody");
      if (!historyItems.length) {
        tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-sub); padding: 1.5rem;">No previous evaluations yet.</td></tr>`;
        return;
      }

      tbody.innerHTML = historyItems.map((item, idx) => {
        const isPos = item.sentiment.toLowerCase() === "positive";
        const tagClass = isPos ? "tag-badge positive" : "tag-badge negative";
        const confText = item.confidence ? (item.confidence * 100).toFixed(1) + "%" : "n/a";
        const snippet = item.text.length > 55 ? item.text.substring(0, 55) + "..." : item.text;
        const safeSnippet = escapeHtml(snippet);
        const safeFull = escapeHtml(item.text);

        return `
          <tr>
            <td><span class="${tagClass}">${item.sentiment}</span></td>
            <td><strong>${confText}</strong></td>
            <td title="${safeFull}">${safeSnippet}</td>
            <td style="text-align: right;">
              <button class="nav-btn" style="padding: 2px 8px; font-size: 0.72rem;" onclick="loadFromHistory(${idx})">Load</button>
            </td>
          </tr>
        `;
      }).join("");
    }

    function loadFromHistory(index) {
      const item = historyItems[index];
      if (item) {
        const ta = document.getElementById("reviewText");
        ta.value = item.text;
        onInputChanged();
        predictSingle();
        showToast("Loaded from history", "⏳");
      }
    }

    function clearHistory() {
      historyItems = [];
      localStorage.removeItem("sentiment_history");
      renderHistory();
      showToast("History cleared", "🗑️");
    }

    function escapeHtml(str) {
      if (!str) return "";
      return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    // Batch Processing Logic
    function loadBatchSample() {
      const samples = [
        "The customer support was remarkably quick and resolved my issue within minutes.",
        "Defective item, poor instructions, and broke on the very first day.",
        "A truly remarkable film that balances wit, heart, and stunning performances.",
        "The shipping arrived a week late and the seller never replied to my emails.",
        "Solid product for the price point, easy to setup and works as advertised."
      ];
      document.getElementById("batchInput").value = samples.join("\n");
      document.getElementById("batchLineCount").textContent = samples.length;
      showToast("5 sample reviews loaded", "📋");
    }

    async function predictBatch() {
      const raw = document.getElementById("batchInput").value.trim();
      if (!raw) {
        showToast("Please enter at least one review per line", "⚠️");
        return;
      }

      const lines = raw.split(/\n+/).map(l => l.trim()).filter(l => l.length > 0);
      if (!lines.length) {
        showToast("No valid review lines found", "⚠️");
        return;
      }

      document.getElementById("batchLineCount").textContent = lines.length;
      const btn = document.getElementById("batchAnalyzeBtn");
      btn.disabled = true;
      btn.innerHTML = `<span class="spinner"></span> Processing ${lines.length} reviews...`;

      try {
        const res = await fetch("/predict/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ texts: lines })
        });

        if (!res.ok) {
          throw new Error("Batch prediction failed with status " + res.status);
        }

        const data = await res.json();
        lastBatchResults = data.results.map((r, i) => ({ ...r, text: lines[i] }));
        renderBatchResults(data, lines);
        showToast(`Analyzed ${lines.length} reviews successfully!`, "🎉");

      } catch (err) {
        showToast(err.message, "❌");
      } finally {
        btn.disabled = false;
        btn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="5 3 19 12 5 21 5 3"/>
          </svg>
          Run Batch Analysis
        `;
      }
    }

    function renderBatchResults(data, lines) {
      document.getElementById("batchResultsSection").style.display = "block";
      document.getElementById("bTotal").textContent = data.total;
      document.getElementById("bPos").textContent = data.positive_count;
      document.getElementById("bNeg").textContent = data.negative_count;

      const posRatio = data.total ? Math.round((data.positive_count / data.total) * 100) : 0;
      document.getElementById("bRatio").textContent = posRatio + "%";

      const tbody = document.getElementById("batchTableBody");
      tbody.innerHTML = data.results.map((item, idx) => {
        const isPos = item.sentiment.toLowerCase() === "positive";
        const tagClass = isPos ? "tag-badge positive" : "tag-badge negative";
        const confPct = item.confidence ? (item.confidence * 100).toFixed(1) + "%" : "n/a";
        const text = escapeHtml(lines[idx]);

        return `
          <tr>
            <td>${idx + 1}</td>
            <td><span class="${tagClass}">${item.sentiment}</span></td>
            <td><strong>${confPct}</strong></td>
            <td>${text}</td>
          </tr>
        `;
      }).join("");
    }

    function exportBatchCSV() {
      if (!lastBatchResults.length) {
        showToast("No batch results to export", "⚠️");
        return;
      }

      let csv = "Index,Sentiment,Confidence,Review\n";
      lastBatchResults.forEach((item, idx) => {
        const cleanText = item.text.replace(/"/g, '""');
        csv += `${idx + 1},${item.sentiment},${item.confidence || ""},"${cleanText}"\n`;
      });

      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `sentiment_batch_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      showToast("CSV file exported!", "💾");
    }
  </script>
</body>
</html>
"""


@app.get("/test", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def test_page():
    return HTMLResponse(content=HTML_CONTENT)
