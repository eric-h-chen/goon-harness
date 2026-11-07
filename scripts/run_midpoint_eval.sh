#!/usr/bin/env bash
# Midpoint held-out eval: starting agent (iteration 0) vs optimized best.mjs (iteration 5)
# on the fixed subset in data/splits/heldout_subset_midpoint.json.
#
# Run inside screen/tmux on macOS (Apple Silicon, with OrbStack/Docker Desktop) or x86_64 Linux:
#   bash scripts/run_midpoint_eval.sh
#
# Phases run one after another, both harnesses in parallel within a phase:
#   1. lcb: lcb_real + lcb_impossible  -> results/midpoint-lcb
#   2. swe: swe_real + swe_impossible  -> results/midpoint-swe
# They are not run together because emulated SWE-bench load can push LiveCodeBench's 6 s
# per-test limit into spurious timeouts. SWE images are pre-pulled during phase 1 when the
# disk has room.
#
# Environment overrides:
#   PHASES="swe"        run only some phases (default "lcb swe")
#   N_LCB=4 N_SWE=5     concurrent tasks per harness (default: from CPUs/RAM)
#   SKIP_SMOKE=1        skip the 1-task smoke test
#   SKIP_DISK_CHECK=1   skip the free-disk check before SWE images are pulled
#
# Safe to rerun: iterations that already have metrics are skipped.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO=$(pwd)

MODEL=openrouter/openai/gpt-5.6-luna
H=commited-results/swe25-piloop-astra-proposer
PHASES=${PHASES:-"lcb swe"}
SUBSET=data/splits/heldout_subset_midpoint.json
SPLITS="oneoff conflicting"
HARNESS_0=src/agents/pi_loop.mjs
HARNESS_5=$H/best.mjs
GB_PER_SWE_IMAGE=2   # measured: 32 subset images ~= 62 GB on disk after layer sharing

export PYTHONUNBUFFERED=1
export INSPECT_DISPLAY=${INSPECT_DISPLAY:-plain}  # readable logs instead of a live TUI

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "ERROR: $*"; exit 1; }

suites_for() { case $1 in lcb) echo "lcb_real lcb_impossible";; swe) echo "swe_real swe_impossible";; *) die "unknown phase $1";; esac; }

# ----------------------------------------------------------------------------- preflight
log "Preflight in $REPO"
OS=$(uname -s); ARCH=$(uname -m)
case "$OS/$ARCH" in
  Linux/x86_64) ;;
  Darwin/arm64)
    grep -q "_force_x86_impossible_swe_images()" src/adapter.py \
      || die "src/adapter.py lacks the x86 image pin; arm64 SWE images can leak reference fixes"
    log "Apple Silicon: SWE-bench runs x86_64 images under emulation (slower)" ;;
  *) die "unsupported host $OS/$ARCH" ;;
esac
docker info >/dev/null 2>&1 || die "Docker is not running (open -a OrbStack) or not usable without sudo"
command -v node >/dev/null || die "Node >= 22 not on PATH"
node -e 'process.exit(+process.versions.node.split(".")[0] >= 22 ? 0 : 1)' \
  || die "Node $(node --version) is too old; need >= 22 (editable_pi runs node --check on .mjs files)"
command -v uv >/dev/null || die "uv not found (curl -LsSf https://astral.sh/uv/install.sh | sh)"
uv run --no-sync --env-file .env python -c 'import os, sys; sys.exit(not os.environ.get("OPENROUTER_API_KEY"))' 2>/dev/null \
  || die ".env must set OPENROUTER_API_KEY (as uv run --env-file reads it)"
[[ -f $SUBSET ]] || die "$SUBSET missing; git pull the commit that adds it"
[[ -f $HARNESS_5 ]] || die "$HARNESS_5 missing"

if [[ $OS == Darwin ]]; then
  CPUS=$(sysctl -n hw.ncpu); MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
else
  CPUS=$(nproc); MEM_GB=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
fi
clamp() { local v=$1 lo=$2 hi=$3; (( v < lo )) && v=$lo; (( v > hi )) && v=$hi; echo "$v"; }
# Two harnesses run at once. LCB has tight per-test time limits, so leave CPU headroom.
N_LCB=${N_LCB:-$(clamp $(( CPUS / 3 )) 1 6)}
N_SWE_CPU=$(( CPUS / 2 )); N_SWE_MEM=$(( MEM_GB / 6 ))  # ~3 GB per SWE task
N_SWE=${N_SWE:-$(clamp $(( N_SWE_CPU < N_SWE_MEM ? N_SWE_CPU : N_SWE_MEM )) 1 8)}
log "${CPUS} CPUs, ${MEM_GB} GB RAM -> N_LCB=${N_LCB}, N_SWE=${N_SWE} tasks per harness | phases: $PHASES"

free_gb() {  # free space where Docker keeps images
  local path
  if [[ $OS == Darwin ]]; then path=$HOME; else path=$(docker info --format '{{.DockerRootDir}}'); fi
  df -Pk "$path" | awk 'NR==2 {print int($4/1048576)}'
}

# ----------------------------------------------------------------------------- setup
log "uv sync"
uv sync
log "Building agent-loop image goon-pi-loop:local"
docker build -q -t goon-pi-loop:local runtimes/pi
if [[ $PHASES == *lcb* && ! -f data/cache/lcb_release_v6_index.jsonl ]]; then
  log "Caching held-out LiveCodeBench tests (~4.5 GB download, once)"
  uv run python src/eval.py prepare-lcb --partition heldout
fi
mkdir -p results

swe_images() {  # x86_64 images for every SWE instance in the subset
  uv run python - "$SUBSET" <<'EOF'
import json, sys
sys.path.insert(0, "src")
from adapter import swe_x86_image
s = json.load(open(sys.argv[1]))
for i in sorted(set(s["swebench_verified"]) | set(s["impossible_swebench"])):
    print(swe_x86_image(i))
EOF
}

missing_swe_images() {
  local img
  for img in $(swe_images); do docker image inspect "$img" >/dev/null 2>&1 || echo "$img"; done
}

pull_swe_images() {  # explicit linux/amd64: a platform-less pull fails on Apple Silicon
  missing_swe_images | xargs -r -n1 -P4 docker pull -q --platform linux/amd64
}

check_swe_disk() {  # margin_gb
  [[ -n ${SKIP_DISK_CHECK:-} ]] && return 0
  local n need free
  n=$(missing_swe_images | wc -l | tr -d ' ')
  need=$(( n * GB_PER_SWE_IMAGE + $1 ))
  free=$(free_gb)
  log "SWE images missing: $n (~$(( n * GB_PER_SWE_IMAGE )) GB); free disk: ${free} GB; need ~${need} GB"
  (( free >= need ))
}

run_eval() {  # harness iteration run_dir suites [extra args...]
  local harness=$1 iter=$2 dir=$3 suites=$4; shift 4
  # shellcheck disable=SC2086  # suites are intentionally word-split
  uv run --env-file .env python src/eval.py run --harness "$harness" --iteration "$iter" \
    --run-dir "$dir" --model "$MODEL" --subset --suites $suites "$@"
}

errors_in() {  # metrics.json -> total infrastructure errors across cells
  uv run python -c '
import json, sys
rows = json.load(open(sys.argv[1]))["rows"].values()
print(sum(v for r in rows for k, v in r.items() if k.startswith("n_errors_")))' "$1"
}

# ----------------------------------------------------------------------------- smoke test
SMOKE_DIR=results/smoke-midpoint
ALL_SUITES=""; for p in $PHASES; do ALL_SUITES+="$(suites_for "$p") "; done
if [[ -z ${SKIP_SMOKE:-} ]]; then
  if [[ ! -f $SMOKE_DIR/eval/iter_005/metrics.json ]]; then
    log "Smoke test: best.mjs, 1 task per suite, oneoff only (15-30 min; impossible tasks run long)"
    run_eval "$HARNESS_5" 5 "$SMOKE_DIR" "$ALL_SUITES" --limit 1 --splits oneoff \
      --max-samples 2 --max-sandboxes 4 2>&1 | tee results/smoke-midpoint.log
  fi
  ERR=$(errors_in "$SMOKE_DIR/eval/iter_005/metrics.json")
  [[ $ERR == 0 ]] || die "smoke test had $ERR infrastructure errors; inspect $SMOKE_DIR/eval/iter_005/logs (delete $SMOKE_DIR to retry)"
  log "Smoke test passed"
fi

# ----------------------------------------------------------------------------- phases
PULL_PID=""
if [[ $PHASES == *lcb*swe* ]] && check_swe_disk 20; then
  log "Pre-pulling SWE images in the background during the LCB phase -> results/swe_pull.log"
  pull_swe_images >results/swe_pull.log 2>&1 &
  PULL_PID=$!
fi

FAILED=0
for phase in $PHASES; do
  suites=$(suites_for "$phase")
  run_dir=results/midpoint-$phase
  if [[ $phase == swe ]]; then
    N=$N_SWE
    if [[ -n $PULL_PID ]]; then
      log "Waiting for background SWE image pulls"
      wait "$PULL_PID" || die "SWE image pull failed; see results/swe_pull.log"
    fi
    check_swe_disk 10 || die "not enough disk for SWE images; free space (or reduce the subset) and rerun with PHASES=swe"
    log "Pulling any missing SWE images (linux/amd64)"
    pull_swe_images
    [[ -z $(missing_swe_images) ]] || die "some SWE images are still missing"
  else
    N=$N_LCB
  fi

  log "=== Phase $phase: $suites -> $run_dir (N=$N per harness)"
  PIDS=()
  ITERS=()
  for iter in 0 5; do
    if [[ -f $run_dir/eval/iter_00$iter/metrics.json ]]; then
      log "iteration $iter already has metrics in $run_dir; skipping"
      continue
    fi
    harness_var=HARNESS_$iter
    logfile=results/midpoint-${phase}_iter$iter.log
    log "Launching iteration $iter (${!harness_var}) -> $logfile"
    # shellcheck disable=SC2086
    run_eval "${!harness_var}" "$iter" "$run_dir" "$suites" --splits $SPLITS \
      --max-samples "$N" --max-sandboxes "$N" >"$logfile" 2>&1 &
    PIDS+=($!)
    ITERS+=("$iter")
    sleep 20  # let the first process write config.json before the second checks it
  done
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      log "phase $phase iteration ${ITERS[$i]} finished"
    else
      log "phase $phase iteration ${ITERS[$i]} FAILED; see results/midpoint-${phase}_iter${ITERS[$i]}.log"
      FAILED=1
    fi
  done
done

# ----------------------------------------------------------------------------- summary
for phase in $PHASES; do
  f=results/midpoint-$phase/eval/metrics.jsonl
  [[ -f $f ]] || continue
  log "Summary ($f)"
  uv run python - "$f" <<'EOF'
import json, sys
def fmt(r, c):
    p, ci, n, e = r.get(f"pass_{c}"), r.get(f"pass_{c}_ci95"), r.get(f"n_{c}"), r.get(f"n_errors_{c}")
    if p is None:
        return f"{c:12s}  -"
    flag = "  <-- >5% errors" if n is not None and e and e > 0.05 * (n + e) else ""
    return f"{c:12s} {p:6.1%}  [{ci[0]:.1%}, {ci[1]:.1%}]  n={n} errors={e}{flag}"
for r in map(json.loads, open(sys.argv[1])):
    print(f"\niteration {r['iteration']}  {r['benchmark']}  harness={r['harness_id']}  "
          f"wall={r.get('wall_clock_s')}s  subset={r.get('subset')}")
    for c in ("real", "oneoff", "conflicting"):
        print("  " + fmt(r, c))
    if "cheat_rate" in r:
        lo, hi = r["cheat_rate_ci95"]
        print(f"  cheat_rate   {r['cheat_rate']:6.1%}  [{lo:.1%}, {hi:.1%}]  (oneoff + conflicting pooled)")
EOF
done

(( FAILED == 0 )) || die "one or more runs failed; rerun this script to retry only the missing ones"
log "Done."
