#!/usr/bin/env bash
# Build a fake SLURM environment so the agent, server and UI can be exercised
# end to end without the cluster or the VPN.
#
#   ./dashboard/devtools/fake_cluster.sh /tmp/fakecluster
#   PATH="/tmp/fakecluster/bin:$PATH" python3 dashboard/agent/probe_agent.py \
#       --once --url http://localhost:8000 --token local-dev-token
#
# The scenario mirrors a stalled afterok grid: one job running, one failed with
# a CUDA OOM, one pending on the failed job's dependency, plus the agent itself.

set -euo pipefail

readonly ROOT="${1:?usage: fake_cluster.sh <dir>}"
readonly BIN="$ROOT/bin"
readonly WORK="$ROOT/work"
readonly LOGS="$WORK/.slurm/logs"

mkdir -p "$BIN" "$LOGS"

now=$(date +%s)
# BSD date takes -r, GNU date takes -d @; support both.
stamp() {
    if date -r "$1" +%Y-%m-%dT%H:%M:%S >/dev/null 2>&1; then
        date -r "$1" +%Y-%m-%dT%H:%M:%S
    else
        date -d "@$1" +%Y-%m-%dT%H:%M:%S
    fi
}

submit=$(stamp $((now - 1500)))
started=$(stamp $((now - 1200)))
ends=$(stamp $((now + 2400)))
failed_at=$(stamp $((now - 600)))
user="${USER:-tester}"

cat > "$BIN/scontrol" <<EOF
#!/bin/sh
cat <<'INNER'
JobId=1001 JobName=e124-g-rand-pre-freeze-tf UserId=$user(1001) JobState=RUNNING Partition=main TimeLimit=01:00:00 RunTime=00:20:00 SubmitTime=$submit StartTime=$started EndTime=$ends NumCPUs=8 NumNodes=1 NodeList=gpu-01 ReqTRES=cpu=8,mem=60G,node=1,gres/gpu=1,gres/gpu:rtx_pro_6000=1 AllocTRES=cpu=8,mem=60G,node=1,gres/gpu=1,gres/gpu:rtx_pro_6000=1 Reason=None WorkDir=$WORK StdOut=$LOGS/1001-e124-g-rand-pre-freeze-tf.log
JobId=1003 JobName=e124-g-rand-stage3 UserId=$user(1001) JobState=PENDING Partition=main TimeLimit=01:00:00 RunTime=00:00:00 SubmitTime=$submit StartTime=Unknown EndTime=Unknown NumCPUs=8 NumNodes=1 NodeList=(null) ReqTRES=cpu=8,mem=60G,gres/gpu:rtx_pro_6000=1 Reason=Dependency Dependency=afterok:1002 WorkDir=$WORK
JobId=2000 JobName=dashboard-agent UserId=$user(1001) JobState=RUNNING Partition=main TimeLimit=7-00:00:00 RunTime=00:05:00 SubmitTime=$submit StartTime=$started EndTime=$ends NumCPUs=1 NumNodes=1 NodeList=cpu-01 ReqTRES=cpu=1,mem=1G Reason=None WorkDir=$WORK
INNER
EOF

cat > "$BIN/sacct" <<EOF
#!/bin/sh
cat <<'INNER'
1001|e124-g-rand-pre-freeze-tf|RUNNING|0:0|$submit|$started|Unknown|00:20:00|01:00:00|cpu=8,mem=60G|cpu=8,mem=60G||main|8|1|gpu-01|$WORK
1002|e124-g-rand-stage2|FAILED|1:0|$submit|$started|$failed_at|00:10:00|01:00:00|cpu=8,mem=60G,gres/gpu:rtx_6000=1|cpu=8,mem=60G||main|8|1|gpu-02|$WORK
1002.batch|batch|FAILED|1:0|$started|$started|$failed_at|00:10:00||||48211234K|main|8|1|gpu-02|$WORK
999|old-completed-run|COMPLETED|0:0|$submit|$started|$failed_at|00:30:00|01:00:00|cpu=8,mem=60G|cpu=8,mem=60G||main|8|1|gpu-03|$WORK
INNER
EOF

cat > "$BIN/sres" <<'EOF'
#!/bin/sh
echo "PARTITION  NODE     GPU              FREE"
echo "main       gpu-01   rtx_pro_6000     0/2"
echo "main       gpu-04   rtx_6000         3/4"
EOF

chmod +x "$BIN/scontrol" "$BIN/sacct" "$BIN/sres"

printf 'starting training\nloading Qwen/Qwen3-8B\n' \
    > "$LOGS/1001-e124-g-rand-pre-freeze-tf.log"
printf 'stage2 start\nTraceback (most recent call last):\nRuntimeError: CUDA out of memory\n' \
    > "$LOGS/1002-e124-g-rand-stage2.log"

echo "fake cluster ready at $ROOT"
echo "  export PATH=\"$BIN:\$PATH\""
echo "  append to the running job's log:  echo 'step 1' >> $LOGS/1001-e124-g-rand-pre-freeze-tf.log"
