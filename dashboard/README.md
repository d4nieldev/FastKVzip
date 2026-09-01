# SLURM experiment dashboard

A web dashboard for the jobs in this repo's experiments, reachable from any
device **without the BGU VPN**.

The cluster allows no inbound connections from outside the VPN, so nothing here
connects *to* BGU. Instead a small CPU job runs inside the cluster and pushes
outward over HTTPS to a public server, which serves the UI.

```
BGU cluster (no inbound access)          Public PaaS               Any device
┌──────────────────────────────┐        ┌──────────────┐         ┌──────────┐
│ dashboard-agent (CPU sbatch) │        │ FastAPI      │         │ React UI │
│  scontrol / sacct ─┐         │ HTTPS  │  /api/ingest │◄────────┤ (served  │
│  read log tails ───┼─ POST ──┼───────►│  SQLite +    │  polls  │  by same │
│  self-resubmit     │         │  gzip  │  log files   │────────►│  app)    │
└──────────────────────────────┘        └──────────────┘         └──────────┘
```

## What it shows

- Running, pending, failed and completed jobs, newest job id first.
- Per job: state, requested resources (GPU, memory, CPUs, partition), wall time
  used against the limit, and time remaining.
- `Reason` on pending jobs — which is how a stalled `afterok` chain from
  `slurm/submit_graph_grid.sh` becomes visible: downstream jobs sit in `PENDING`
  with `Reason=Dependency` after an early failure.
- The complete log of each job, tailing live, with an errors-only filter and a
  full download.
- A time window (1h / 6h / 24h / 7d / 30d) selecting jobs that were **running or
  stopped inside that window**.
- Failed and finished jobs can be dismissed from the view, and restored later.
- The latest `sres` GPU availability as a searchable table, each node scored on
  how free it is and tinted red to green to match.

## Security

Read access is **not** authenticated: anyone with the URL can see your job
names, file paths and logs. Only ingest is gated, by a shared token, so an
outsider cannot poison the job list. Deploy it on an unguessable URL, and if you
later want the read side gated too, that is a small change to `server/app/main.py`.

Nothing the dashboard does can affect the cluster. There is no `scancel` and no
resubmission of your jobs; "dismiss" only hides a record from this dashboard.

## 1. Deploy the server

One Dockerfile, so Railway, Render, Fly or anything else that builds a container
will do.

```bash
cd dashboard
docker build -t slurm-dashboard .
```

Environment:

| Variable | Meaning |
|---|---|
| `DASHBOARD_TOKEN` | **Required.** Shared secret the agent presents on ingest. |
| `DATA_DIR` | Where SQLite and the logs live. Default `/data`. |
| `RETENTION_DAYS` | Jobs are pruned this long after they end. Default `30`. |
| `PORT` | Injected by most platforms; defaults to `8000`. |

Mount a persistent volume at `DATA_DIR` if the platform offers one. If it does
not, nothing breaks: the server tells the agent which logs it is missing and the
agent re-ships them from the beginning on the next poll.

Check it: `curl https://your-app.example.com/healthz`.

## 2. Start the agent on the cluster

```bash
cd /home/danieloh/FastKVzip-implicit
git pull

mkdir -p ~/.fastkvzip-dashboard .slurm/logs
cp dashboard/agent/env.example ~/.fastkvzip-dashboard/env
$EDITOR ~/.fastkvzip-dashboard/env      # set DASHBOARD_URL and DASHBOARD_TOKEN
chmod 600 ~/.fastkvzip-dashboard/env

sbatch dashboard/agent/dashboard_agent.sbatch
```

The token is read from that file rather than passed with `--export`, because
anything in `--export` is visible to every user through `scontrol show job`.

Before submitting for real, confirm the agent parses your cluster's output:

```bash
python3 dashboard/agent/probe_agent.py --once --dry-run
```

That prints the payload it would send. No network, no writes, no state changes.
Compare a few fields against `scontrol show job <id>` and `sacct -j <id>`.

The agent needs no virtualenv — it is standard library only, deliberately, so
that upgrading the research environment can never take monitoring down.

### It keeps itself alive

Roughly ten minutes before its wall time expires the agent submits its own
successor with `--dependency=afterany:$SLURM_JOB_ID` and exits. The dashboard
goes briefly stale during the handover and recovers on its own.

If the agent dies for real, the UI says so in red at the top of the page — that
banner is the one thing to trust, because everything below it is frozen history
once the agent stops reporting.

## Local development

```bash
# Server, with auto-reload
cd dashboard/server
pip install -r requirements-dev.txt
DATA_DIR=./data DASHBOARD_TOKEN=local-dev-token uvicorn app.main:app --reload

# UI, proxying /api to the server above
cd dashboard/web
npm install
npm run dev            # http://localhost:5173
```

`npm run build` writes into `server/static/`, which the server serves directly —
one URL, no CORS.

### Without a cluster

`devtools/fake_cluster.sh` stubs `scontrol`, `sacct` and `sres` with a scenario
that mirrors a stalled `afterok` grid: one job running, one failed on a CUDA
OOM, one pending on the failed job's dependency, plus the agent itself.

```bash
./dashboard/devtools/fake_cluster.sh /tmp/fakecluster
export PATH="/tmp/fakecluster/bin:$PATH"
python3 dashboard/agent/probe_agent.py --once \
    --url http://localhost:8000 --token local-dev-token

# then watch the log tail update live
echo "step 1" >> /tmp/fakecluster/work/.slurm/logs/1001-*.log
```

### Tests

```bash
cd dashboard/server && pytest tests   # agent parsing, ingest, API
cd dashboard/web && npm test          # log line splitting and collapsing
```

Covers SLURM output parsing, log offset handling (append, duplicate, gap,
rewind, in-place rewrite, wiped server), the time-window query, dismissal
surviving re-ingest, retention, and the HTTP surface.

## How the availability score works

Every `sres` row is scored in [0, 1] and tinted from red to green:

```
score = gpu^0.6 * mem^0.2 * cpu^0.2
```

each term being that resource's free fraction. Resources `sres` does not report
are dropped and the remaining exponents renormalised, so output that lists only
GPU counts is scored on GPUs alone rather than read as two resources at zero.

A weighted geometric mean rather than an average, for one reason: any resource
at zero has to take the whole score to zero. A node with every GPU busy is
useless for these jobs however much memory sits idle, and an average would score
exactly that node 0.67 and paint it green. The GPU carries the largest weight
because it is what these jobs actually queue for.

The score measures headroom, not fit -- a node with one of four GPUs free scores
0.25, though a single-GPU job would run on it perfectly well.

Colour is never the whole answer: each row keeps its own `free/total` counts and
states its score as a number and a bar length, so the ranking survives red-green
colour blindness and greyscale.

Nothing is hard-coded to a column layout, because `sres` is site-local with no
stable documented format. Any column whose cells read as `free/total` is taken
for a resource, named by its own header or by the header of the column naming it
(`GPU  FREE`). When no column can be identified the panel falls back to the raw
output, which the "Raw" button also shows on demand.

## How the log transfer works

Logs from these runs reach hundreds of megabytes, mostly tqdm progress lines, so
they are never re-uploaded whole. The agent tracks a byte offset per job and
ships only what is new, capped at 512 KB per job per poll; the server appends it
to a plain file. The result is the complete log on the server, at the cost of
the bytes actually written.

The server, not the agent, is the authority on how much of each log it holds. It
answers every poll with the true offsets, so the two re-align by themselves
after a lost response, a wiped disk, or a job that rewrote its log in place.
