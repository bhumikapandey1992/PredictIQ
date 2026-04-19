# PredictIQ — Predictive Analytics Dashboard

End-to-end ML pipeline with data ingestion, feature engineering, model training, and an interactive dashboard — across three real-world domains.

## Domains

| Domain | Task | Data Source | Target |
|--------|------|-------------|--------|
| **Stocks** | Classification | yfinance (live market data) | Next-day price direction (Up / Down) |
| **Sales** | Regression | Synthetic retail data | Predicted sales units |
| **Healthcare** | Classification | Sklearn breast cancer dataset | Diagnosis (Benign / Malignant) |

## Features

- **Data Explorer** — raw data preview, summary statistics, domain-specific charts
- **Model Performance** — accuracy/F1/RMSE/R² metrics, feature importance bar chart
- **Predict** — interactive input form with real-time prediction and probability chart
- Switch between **XGBoost** and **LightGBM** with one click
- Fully containerized — runs with a single command

## Stack

| Layer | Technology |
|-------|-----------|
| ML Models | XGBoost, LightGBM |
| Feature Engineering | Pandas, Scikit-learn |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Data Sources | yfinance, sklearn datasets, synthetic |
| Containerization | Docker + Docker Compose |

## Getting Started

**Prerequisites:** Docker and Docker Compose installed.

```bash
git clone https://github.com/bhumikapandey1992/PredictIQ.git
cd PredictIQ
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Frontend Dashboard | http://localhost:8502 |
| Backend API | http://localhost:8001 |
| API Health | http://localhost:8001/health |

## Usage

1. Select a **domain** (Stocks / Sales / Healthcare) in the sidebar
2. Choose a **model** (XGBoost or LightGBM)
3. For Stocks, enter a ticker symbol and historical period
4. Click **Ingest Data** → then **Train Model**
5. Explore the **Data Explorer**, **Model Performance**, and **Predict** tabs

## Project Structure

```
PredictIQ/
├── backend/
│   ├── main.py                  # FastAPI — /ingest, /train, /predict, /metrics, /data
│   └── pipeline/
│       ├── ingest.py            # Data loading (yfinance, synthetic, sklearn)
│       ├── features.py          # Feature engineering per domain
│       └── train.py             # XGBoost / LightGBM training + evaluation
├── frontend/
│   └── app.py                   # Streamlit dashboard
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt             # Backend dependencies
└── requirements.frontend.txt   # Frontend dependencies
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest/{domain}` | Load and cache domain data |
| `POST` | `/train/{domain}?model_type=xgboost` | Train model, return metrics + importance |
| `POST` | `/predict/{domain}` | Single prediction from feature dict |
| `GET` | `/data/{domain}` | Retrieve cached data + stats |
| `GET` | `/metrics/{domain}` | Retrieve stored metrics + importance |
| `GET` | `/health` | Health check |
