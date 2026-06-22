# Presentation outline (15-20 slides)

1. Title — Bloc 4 Skin Lesion MLOps Solution
2. Problem statement and business need (triage support)
3. Cahier des charges — functional and non-functional requirements
4. Dataset and preprocessing pipeline
5. Class distribution and data balancing strategy
6. Model architecture — ResNet18 transfer learning
7. Training metrics — accuracy, F1, AUC, confusion matrix
8. System architecture diagram — Docker Compose + K8s
9. MinIO and MLflow artifact management
10. FastAPI serving and Swagger demo
11. CI pipeline — tests, Docker build, manifest validation
12. Monitoring — Evidently / drift report screenshots
13. Automated retraining workflow
14. GDPR, ethics, and limitations
15. Demo results and next steps

# 5-minute video script

| Time | Action |
|------|--------|
| 0:00-1:00 | `docker compose up -d` then open `/docs` and run `/predict` |
| 1:00-2:00 | Show MLflow UI run + MinIO bucket with model artifact |
| 2:00-3:00 | Open GitHub Actions CI run (green pipeline) |
| 3:00-4:00 | Run `python monitoring/drift_report.py` and open drift HTML |
| 4:00-5:00 | Trigger retrain workflow, show promoted model and recap disclaimer |

# Speaker notes

- Emphasize stratified split before oversampling (no leakage)
- Mention fallback to `models/best_model.pt` when MLflow registry is unavailable
- Clarify that monitoring uses prediction distribution drift as a practical proxy
