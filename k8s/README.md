# Kubernetes deployment (minikube / kind)

## Prerequisites

- Docker Desktop
- `kubectl`
- `minikube`

## Deploy

```bash
cp k8s/secrets.yaml.example k8s/secrets.yaml
bash scripts/deploy_k8s.sh
```

## Access API

```bash
minikube service skin-api -n skin-mlops --url
```

Or open `http://localhost:30080/docs` when using NodePort with minikube tunnel.

## Validate manifests in CI

```bash
kubectl apply --dry-run=client -f k8s/
```
