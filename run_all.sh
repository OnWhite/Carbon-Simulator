#!/usr/bin/env bash
#
# run_all.sh -- launch the full verification stack: DP (V1), RL (V2), MARL (V3).
#
#   ./run_all.sh test     # fast: unit + parity tests only, blocking
#   ./run_all.sh dp       # V1 standalone DP solve / horizon probe, blocking
#   ./run_all.sh rl       # V2 single-agent PPO vs exact DP, 2 seeds, background
#   ./run_all.sh marl     # V3 multi-agent, 4 runs on GPU 0-3, background
#   ./run_all.sh all      # test -> dp -> rl -> marl
#   ./run_all.sh status   # one-line status per run
#   ./run_all.sh kill     # stop everything this script started
#
# Run from the repo root. Every tier writes to rllib/exp/<run>/stdout.log.

set -euo pipefail
cd "$(dirname "$0")"

EXP=rllib/exp
MARL_RUNS=(verify_single_s1 verify_multi_s1 verify_single_s2 verify_multi_s2)
RL_RUNS=(verify_v2_s1 verify_v2_s2)
ALL_RUNS=("${RL_RUNS[@]}" "${MARL_RUNS[@]}")
PIDFILE=.run_all.pids

# Tunables -- override from the environment, e.g. HORIZON=12 ./run_all.sh rl
HORIZON="${HORIZON:-10}"
MAX_HOURS="${MAX_HOURS:-8}"
RL_WORKERS="${RL_WORKERS:-2}"

# Ray must not spill to /tmp: on the cluster / is a shared 1.8T disk that
# other users keep at ~100%, and a raylet that cannot spill dies mid-run.
# But the replacement must also be SHORT. Ray builds
#   $RAY_TMPDIR/ray/session_<26-char stamp>_<pid>/sockets/plasma_store
# which adds ~68 chars, against a hard AF_UNIX limit of 107. Pointing
# RAY_TMPDIR at the repo (66 chars) satisfied "not /tmp" and still killed
# both train_v2 runs at ray.init(), 20 minutes into the DP solve:
#   OSError: AF_UNIX path length cannot exceed 107 bytes
# training_script.py survived only because it overrides RAY_TMPDIR itself
# with BASE=/scratch/$USER. Use the same base here so every tier agrees.
# Pick the first base we can actually create a directory in. Testing
# `-w /scratch` was wrong: on the cluster /scratch is root-owned
# drwxrwxr-x, so the test fails even though /scratch/$USER is ours and
# writable. That sent RAY_TMPDIR to $HOME on the 100%-full / and the
# disk check then refused to start at all.
RAY_SPILL=""
if [ -z "${RAY_TMPDIR:-}" ]; then
  for _cand in "/scratch/$USER" "$HOME"; do
    if mkdir -p "$_cand/ray_tmp" 2>/dev/null && [ -w "$_cand/ray_tmp" ]; then
      RAY_TMPDIR="$_cand/ray_tmp"
      RAY_SPILL="$_cand/ray_spill"
      break
    fi
  done
fi
[ -n "${RAY_TMPDIR:-}" ] || die "no writable base for ray's temp dir (tried /scratch/$USER and $HOME)."
[ -n "$RAY_SPILL" ] || RAY_SPILL="$(dirname "$RAY_TMPDIR")/ray_spill"
export RAY_TMPDIR
export TMPDIR="${TMPDIR_OVERRIDE:-$RAY_TMPDIR}"
mkdir -p "$RAY_TMPDIR" "$RAY_SPILL"

# 107 - 68 = 39 usable chars for the base. Checked before anything runs,
# because the failure surfaces only after the DP solve.
[ "${#RAY_TMPDIR}" -le 39 ] || die "RAY_TMPDIR is ${#RAY_TMPDIR} chars; ray's sockets need <= 39 (107-byte AF_UNIX limit minus ~68 for session+socket). Set RAY_TMPDIR to something shorter."

log()  { printf '\033[1m[run_all]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[run_all] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

preflight() {
  local branch; branch="$(git rev-parse --abbrev-ref HEAD)"
  log "branch: $branch    commit: $(git rev-parse --short HEAD)"

  # train_v2.py and the verify_* configs only exist on verification-rebuild.
  [ -f rllib/RL/train_v2.py ] \
    || die "rllib/RL/train_v2.py missing. You are on '$branch'; it lives on verification-rebuild."

  for d in "${MARL_RUNS[@]}"; do
    [ -f "$EXP/$d/config.yaml" ] || die "$EXP/$d/config.yaml missing."
  done

  # Mirror wandb_auth() exactly: env var, else a ~/.netrc that actually
  # PARSES. Grepping for the hostname is not enough -- a netrc with stray
  # text in it (e.g. a pasted "chmod 600 ~/.netrc") greps fine but raises
  # in netrc.netrc(), which wandb_auth swallows, and the run then dies
  # minutes later with "W&B credentials not found".
  case "${WANDB_MODE:-}" in
    offline|disabled|dryrun) log "WANDB_MODE=$WANDB_MODE: not syncing" ;;
    *)
      if [ -z "${WANDB_API_KEY:-}" ] && ! python3 - <<'PY' 2>/dev/null
import netrc, sys
a = netrc.netrc().authenticators("api.wandb.ai")
sys.exit(0 if (a and a[2]) else 1)
PY
      then
        die "no usable wandb credentials. Set WANDB_API_KEY, or repair ~/.netrc (it must parse), or run with WANDB_MODE=offline."
      fi
      ;;
  esac

  # Free space on the filesystem that will hold ray's spill directory.
  local avail_kb
  avail_kb="$(df -Pk "$RAY_TMPDIR" | awk 'NR==2 {print $4}')"
  if [ "${avail_kb:-0}" -lt 20971520 ]; then      # 20 GiB
    die "only $((avail_kb / 1048576)) GiB free on $(df -Ph "$RAY_TMPDIR" | awk 'NR==2 {print $6}') -- ray will fail to spill. Free space or point RAY_TMPDIR elsewhere."
  fi
  log "ray tmp/spill: $RAY_TMPDIR ($((avail_kb / 1048576)) GiB free)"

  # A dirty tree means the logged commit does not describe what actually ran.
  if [ -n "$(git status --porcelain --untracked-files=no -- Carbon_simulator rllib)" ]; then
    log "WARNING: uncommitted changes under Carbon_simulator/ or rllib/."
    log "         manifest.json will record a commit that is not what ran."
  fi

  mkdir -p "${ALL_RUNS[@]/#/$EXP/}"
  log "preflight OK"
}

# Records PID so `status` and `kill` can find the run later.
launch() {                      # launch <name> <cuda_devices|-> <cmd...>
  local name="$1"; shift
  local dev="$1"; shift
  local logfile="$EXP/$name/stdout.log"
  mkdir -p "$EXP/$name"
  [ -f "$logfile" ] && mv "$logfile" "$logfile.$(date +%Y%m%d-%H%M%S).bak"
  if [ "$dev" = "-" ]; then
    PYTHONPATH=. nohup "$@" > "$logfile" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$dev" PYTHONPATH=. nohup "$@" > "$logfile" 2>&1 &
  fi
  echo "$name $!" >> "$PIDFILE"
  log "launched $name (pid $!, gpu ${dev}) -> $logfile"
}

tier_test() {
  log "=== tests (blocking) ==="
  PYTHONPATH=. python3 -m unittest rllib.DP.Unittests -v
  PYTHONPATH=. python3 -m unittest rllib.DP.test_parity -v
  log "tests passed"
}

tier_dp() {
  log "=== V1: exact DP (blocking) ==="
  # train_v2.py solves the DP itself per run; this is the standalone
  # solve + state-count probe, and it fails fast if the horizon has
  # blown past what is enumerable.
  mkdir -p "$EXP/verify_dp"
  PYTHONPATH=. python3 -u rllib/DP/probe_horizon.py \
      --min-t "$HORIZON" --max-t "$HORIZON" --solve \
      2>&1 | tee "$EXP/verify_dp/stdout.log"
  log "DP solve done -> $EXP/verify_dp/stdout.log"
}

tier_rl() {
  log "=== V2: single-agent RL vs exact DP (background) ==="
  local seed
  for seed in 1 2; do
    launch "verify_v2_s$seed" - \
      python3 -u rllib/RL/train_v2.py \
        --run_dir "$EXP/verify_v2_s$seed" \
        --horizon "$HORIZON" --seed "$seed" \
        --max-hours "$MAX_HOURS" --num-workers "$RL_WORKERS"
  done
}

tier_marl() {
  log "=== V3: multi-agent (background, GPU 0-3) ==="
  local gpu=0 d
  for d in "${MARL_RUNS[@]}"; do
    launch "$d" "$gpu" python3 -u rllib/training_script.py --run_dir "$EXP/$d"
    gpu=$((gpu + 1))
  done
}

status() {
  printf '%-20s %-8s %s\n' RUN PID LAST
  local name pid state line
  for name in "${ALL_RUNS[@]}"; do
    pid="$(awk -v n="$name" '$1==n {p=$2} END {print p}' "$PIDFILE" 2>/dev/null || true)"
    state="dead"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && state="alive"
    if [ -f "$EXP/$name/stdout.log" ]; then
      # `|| true` matters: under `set -e` with pipefail, a grep that finds
      # nothing (a log that has only just been created) aborted the whole
      # loop, so status printed its header and no rows at all.
      line="$(grep -E 'iter |Traceback|Error|V\*|rel_regret' "$EXP/$name/stdout.log" 2>/dev/null | tail -1 | cut -c1-90 || true)"
      [ -z "$line" ] && line="$(tail -1 "$EXP/$name/stdout.log" 2>/dev/null | cut -c1-90 || true)"
      [ -z "$line" ] && line="(log empty)"
    else
      line="NO LOG"
    fi
    printf '%-20s %-8s %s  %s\n' "$name" "${pid:--}" "[$state]" "$line"
  done
}

kill_all() {
  [ -f "$PIDFILE" ] || die "no $PIDFILE; nothing launched from this script."
  local name pid
  while read -r name pid; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" && log "killed $name ($pid)"; fi
  done < "$PIDFILE"
  rm -f "$PIDFILE"
}

case "${1:-all}" in
  test)   preflight; tier_test ;;
  dp)     preflight; tier_dp ;;
  rl)     preflight; tier_rl;   sleep 5; status ;;
  marl)   preflight; tier_marl; sleep 5; status ;;
  all)    preflight; tier_test; tier_dp; tier_rl; tier_marl; sleep 5; status ;;
  status) status ;;
  kill)   kill_all ;;
  *)      die "unknown tier '${1}'. Use: test | dp | rl | marl | all | status | kill" ;;
esac
