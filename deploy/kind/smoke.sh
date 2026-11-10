#!/usr/bin/env bash
# Exercise the Kubernetes backend on the cluster from up.sh: jobs run as pods
# on the nodes the scheduler picked, a borrowed job is preempted when another
# team reclaims its quota, and it resumes from its checkpoint.
set -euo pipefail

NS=forge
TAG=$RANDOM$RANDOM
api() { local key=$1; shift; kubectl -n $NS exec deploy/forge-api -- env FORGE_API_KEY="$key" "$@" | tr -d '\r'; }
bootstrap() { kubectl -n $NS exec deploy/forge-api -- forge admin bootstrap --team "$1" --quota 8 --user "$1-$TAG" 2>/dev/null | tr -d '\r'; }
state() { api "$1" forge status "$2" | head -1 | awk '{print $NF}'; }
wait_state() { # key, job, wanted state, timeout seconds
  for _ in $(seq 1 "$4"); do
    [ "$(state "$1" "$2")" = "$3" ] && return 0
    sleep 1
  done
  echo "FAIL: job $2 did not reach $3 (is $(state "$1" "$2"))"; api "$1" forge status "$2"
  kubectl -n $NS logs deploy/forge-controller --tail=40
  exit 1
}
TRAIN="python examples/train.py --checkpoint-seconds 5"

VISION=$(bootstrap vision)
PLANNING=$(bootstrap planning)

echo "== two vision jobs fill both nodes; the second runs on borrowed capacity"
A=$(api "$VISION" forge submit --gpus 8 --image forge:dev --name within-quota -- $TRAIN --steps 6000 --sleep 0.05)
wait_state "$VISION" "$A" RUNNING 90
B=$(api "$VISION" forge submit --gpus 8 --image forge:dev --name borrowed --priority low -- $TRAIN --steps 6000 --sleep 0.05)
wait_state "$VISION" "$B" RUNNING 90
kubectl -n forge-vision get pods -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,GPUS:.spec.containers[0].resources.limits.nvidia\.com/gpu,PHASE:.status.phase'
[ "$(kubectl -n forge-vision get pods -o jsonpath='{.items[*].spec.nodeName}' | tr ' ' '\n' | sort -u | wc -l)" -eq 2 ] \
  || { echo "FAIL: expected the two jobs on different nodes"; exit 1; }

echo "== planning reclaims its quota once the borrowed job is past the minimum runtime"
sleep 25
P=$(api "$PLANNING" forge submit --gpus 8 --image forge:dev --name reclaim --priority high -- $TRAIN --steps 300 --sleep 0.02)
wait_state "$VISION" "$B" QUEUED 90
api "$VISION" forge status "$B" | grep -q "PREEMPTING" || { echo "FAIL: borrowed job was not preempted"; exit 1; }
api "$VISION" forge status "$B" | grep -q "checkpoint at step" || { echo "FAIL: no checkpoint was saved"; exit 1; }
[ "$(state "$VISION" "$A")" = RUNNING ] || { echo "FAIL: the within-quota job should not have been touched"; exit 1; }
wait_state "$PLANNING" "$P" SUCCEEDED 120

echo "== the preempted job gets its node back and resumes from the checkpoint"
wait_state "$VISION" "$B" RUNNING 90
for _ in $(seq 1 30); do
  api "$VISION" forge logs "$B" --tail 2000 | grep -q "resumed from step" && break
  sleep 1
done
api "$VISION" forge logs "$B" --tail 2000 | grep "resumed from step" || { echo "FAIL: did not resume from a checkpoint"; exit 1; }

echo "== cancel removes the pods"
api "$VISION" forge cancel "$A" >/dev/null
api "$VISION" forge cancel "$B" >/dev/null
for _ in $(seq 1 60); do
  [ -z "$(kubectl -n forge-vision get pods -o name)" ] && break
  sleep 1
done
[ -z "$(kubectl -n forge-vision get pods -o name)" ] || { echo "FAIL: pods were left behind"; kubectl -n forge-vision get pods; exit 1; }
api "$VISION" forge cluster
echo "PASS"
