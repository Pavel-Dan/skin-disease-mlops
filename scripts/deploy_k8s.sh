#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v minikube >/dev/null 2>&1; then
  echo "minikube is required."
  exit 1
fi

minikube start
eval "$(minikube docker-env)"

docker build -t skin-api:local .

if [ ! -f k8s/secrets.yaml ]; then
  cp k8s/secrets.yaml.example k8s/secrets.yaml
fi

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/minio.yaml
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/api.yaml

kubectl rollout status deployment/minio -n skin-mlops
kubectl rollout status deployment/mlflow -n skin-mlops
kubectl rollout status deployment/skin-api -n skin-mlops

echo "API URL:"
minikube service skin-api -n skin-mlops --url
