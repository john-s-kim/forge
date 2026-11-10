#!/usr/bin/env bash
# Bring up Forge on a local kind cluster, with each worker node advertising
# eight GPUs that do not exist. Needs docker, kind, kubectl and helm.
#
#   deploy/kind/up.sh        then        deploy/kind/smoke.sh
#   kind delete cluster --name forge     to tear down
set -euo pipefail
cd "$(dirname "$0")/../.."

CLUSTER=${CLUSTER:-forge}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

kind get clusters | grep -qx "$CLUSTER" || kind create cluster --name "$CLUSTER" --config deploy/kind/kind.yaml
kubectl config use-context "kind-$CLUSTER" >/dev/null

docker build -q -t forge:dev .
kind load docker-image forge:dev --name "$CLUSTER"

# Advertise an extended resource on the workers. The kubelet accepts it as
# capacity and enforces it at admission; no device plugin is involved.
for node in $(kubectl get nodes -l '!node-role.kubernetes.io/control-plane' -o name); do
  kubectl patch "$node" --subresource=status --type=json \
    -p "[{\"op\":\"add\",\"path\":\"/status/capacity/nvidia.com~1gpu\",\"value\":\"$GPUS_PER_NODE\"}]" >/dev/null
done

kubectl create namespace forge --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl -n forge apply -f deploy/kind/deps.yaml >/dev/null
kubectl -n forge rollout status deploy/postgres deploy/redis deploy/nats --timeout=180s

helm upgrade --install forge deploy/helm/forge -n forge \
  --set image.tag=dev \
  --set secret="${FORGE_SECRET:-kind-dev-secret}" \
  --set 'teams={vision,planning}' \
  --set minRuntimeSeconds=20 \
  --set preemptionGraceSeconds=20 \
  --wait --timeout 180s
kubectl -n forge rollout status deploy/forge-api deploy/forge-controller --timeout=120s
echo "Forge is up. Next: deploy/kind/smoke.sh"
