# Sentiment Analysis API

A production-ready FastAPI service that performs sentiment analysis on textual reviews and feedback using a scikit-learn pipeline (TF-IDF Vectorizer + Logistic Regression).

---

## What the Model Predicts

The model performs **binary sentiment classification**:
- It evaluates input text (e.g., product reviews, customer feedback, comments) and classifies it as either **`positive`** or **`negative`**.
- In addition to the predicted label, it returns the model's **confidence** score and probability distribution (`probabilities` for both classes).

---

## Example Request Body for `/predict`

Endpoint: `POST /predict`  
Header: `Content-Type: application/json`

```json
{
  "text": "This workshop on ML in production is absolutely fantastic!"
}
```

### Example Response

```json
{
  "sentiment": "positive",
  "confidence": 0.7113,
  "probabilities": {
    "negative": 0.2887,
    "positive": 0.7113
  }
}
```

---

## How to Run Locally

You can run the application either directly with Python or via Docker.

### Option 1: Using Python & Uvicorn (Recommended)

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd ml_in_production
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the API server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access the application:**
   - **Interactive Web UI:** [https://ml-workshop-sept-2026.onrender.com/test](https://ml-workshop-sept-2026.onrender.com/test)
   - **Interactive API Docs (Swagger):** [https://ml-workshop-sept-2026.onrender.com/docs](https://ml-workshop-sept-2026.onrender.com/docs)
   - **Health Check:** [https://ml-workshop-sept-2026.onrender.com/health](https://ml-workshop-sept-2026.onrender.com/health)

---

### Option 2: Using Docker

1. **Build the Docker container:**
   ```bash
   docker build -t sentiment-api .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 sentiment-api
   ```

---

### Testing the Endpoint via `curl`

Once the server is running, send a test request:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This workshop on ML in production is absolutely fantastic!"}'
```
