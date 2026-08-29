#!/usr/bin/env bash
# Long-running CPU job that pushes SLURM state to the dashboard server.
# Deliberately skips the FASTKVZIP_VENV guard rails the training scripts use:
# the agent is stdlib-only so monitoring cannot be broken by the research env.
#SBATCH --job-name=dashboard-agent
# The CPU partition: this job only shells out to scontrol/sacct and reads log
# files, so it must never occupy a GPU. (The repo's training scripts say
# "main", but every real job on this cluster runs on "gpu" or "cpu".)
#SBATCH --partition=cpu
# Kept conservative so submission is not rejected by a partition limit. The
# agent resubmits its own successor before this expires, so raising it is an
# optimization (fewer handovers), not a requirement. Check the cap with
#   sinfo -p cpu -o '%P %l'
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=.slurm/logs/%j-%x.log

set -euo pipefail

# SLURM copies the batch script into a spool directory before running it, so
# BASH_SOURCE points at /var/spool/slurmd/... and not at the checkout. Locate
# the real agent directory from where sbatch was invoked instead.
if [[ -n "${DASHBOARD_AGENT_DIR:-}" ]]; then
    AGENT_DIR="$DASHBOARD_AGENT_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    AGENT_DIR="$SLURM_SUBMIT_DIR/dashboard/agent"
else
    AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
readonly AGENT_DIR
# Named literally rather than via basename: under SLURM the running copy is
# called "slurm_script", and the successor must be submitted from this file.
readonly SCRIPT_PATH="$AGENT_DIR/dashboard_agent.sbatch"
readonly ENV_FILE="${DASHBOARD_ENV_FILE:-$HOME/.fastkvzip-dashboard/env}"

if [[ ! -f "$AGENT_DIR/probe_agent.py" ]]; then
    echo "cannot find probe_agent.py in $AGENT_DIR" >&2
    echo "submit from the repository root, or set DASHBOARD_AGENT_DIR to" >&2
    echo "the absolute path of dashboard/agent" >&2
    exit 2
fi

# The token is read from a chmod-600 file rather than --export so it never
# appears in scontrol/squeue output, which any cluster user can read.
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [[ -z "${DASHBOARD_URL:-}" ]]; then
    echo "DASHBOARD_URL is unset. Create $ENV_FILE from dashboard/agent/env.example" >&2
    exit 2
fi
if [[ -z "${DASHBOARD_TOKEN:-}" ]]; then
    echo "DASHBOARD_TOKEN is unset. Create $ENV_FILE from dashboard/agent/env.example" >&2
    exit 2
fi

echo "agent job ${SLURM_JOB_ID:-<none>} on $(hostname), reporting to $DASHBOARD_URL"

exec python3 -u "$AGENT_DIR/probe_agent.py" \
    --script-path "$SCRIPT_PATH" \
    --interval "${DASHBOARD_POLL_INTERVAL:-30}" \
    "$@"
