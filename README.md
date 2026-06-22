# Bloc 4 — Skin Lesion MLOps Solution

End-to-end MLOps pipeline for multi-class skin lesion classification: preprocessing, training, model serving, CI/CD, monitoring, and automated retraining.

## Project scope

- **Task:** classify skin disease images into **10 clinical categories** (Kaggle dataset)
- **Model:** EfficientNet-B3 (transfer learning)
- **Disclaimer:** decision-support tool only — not a medical diagnosis device

## Repository structure

```
notebooks/          EDA and baseline experiments
src/data/           preprocessing pipeline (existing)
src/training/       model training and evaluation
api/                FastAPI inference service
monitoring/         drift detection reports
retrain/            automated retraining entrypoint
k8s/                Kubernetes manifests
scripts/            setup and deployment helpers
tests/              unit and integration tests
.github/workflows/  CI/CD pipelines
```

## Quick start

### 1. Install dependencies

```bash
uv sync
# or
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

### 3. Prepare data (Kaggle → MinIO → local)

Credentials: save your Kaggle API token to `C:\Users\<you>\.kaggle\access_token` or set `KAGGLE_API_TOKEN` in `.env` (never commit).

```bash
# Start MinIO + create buckets
docker compose up -d minio
bash scripts/setup_minio.sh

# Full pipeline (reads token from .env or ~/.kaggle/access_token)
python scripts/run_data_pipeline.py --from-kaggle --purge-raw

# Later runs (data already in MinIO):
python scripts/run_data_pipeline.py --from-minio

# Train (auto-syncs processed data from MinIO if missing locally)
python -m src.training.train --epochs 15
```

Dataset: [ismailpromus/skin-diseases-image-dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)

For CI/local smoke tests without Kaggle:

```bash
python scripts/generate_demo_data.py --processed-only --processed-per-split 8
```

### 4. Train model

```bash
python -m src.training.train --epochs 10
python -m src.training.evaluate
```

### 5. Run API locally

```bash
uvicorn api.main:app --reload --port 8000
```

Open Swagger UI: `http://localhost:8000/docs`

### 6. Docker Compose (MinIO + MLflow + API)

```bash
docker compose up -d --build
bash scripts/setup_minio.sh
```

### 7. Kubernetes (minikube)

```bash
bash scripts/deploy_k8s.sh
```

### 8. Monitoring and retraining

```bash
python monitoring/drift_report.py
python retrain/run.py --force --epochs 3
```

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health |
| `/predict` | POST | Image upload → class probabilities |
| `/model-info` | GET | Model metadata and metrics |

## Evaluation criteria mapping

| Area | Implementation |
|------|----------------|
| Model quality | `src/training/evaluate.py`, F1/AUC/confusion matrix |
| API serving | `api/main.py`, input validation, prediction logging |
| CI/CD | `.github/workflows/ci.yml` |
| Retraining | `retrain/run.py`, `.github/workflows/retrain.yml` |
| Monitoring | `monitoring/drift_report.py` |
| Scalability | stateless API, K8s deployments, MinIO S3 artifacts |
| GDPR / ethics | anonymized images, disclaimer, log retention documented below |

## GDPR and ethics notes

- Images must be anonymized and must not contain patient identifiers
- Prediction logs are stored in `data/logs/predictions.csv` for monitoring only
- Recommended retention: 90 days, then purge logs on request
- The API returns an explicit medical disclaimer on every prediction

## Development commands

```bash
pytest tests/ -q
python scripts/generate_demo_data.py --processed-only
python monitoring/drift_report.py
kubectl apply --dry-run=client -f k8s/
```

## License

Academic project — Bloc 4 AI & ML Solution.
