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

The dashboard opens on two ways in. **Users** is who ran something: everyone
running an agent against this server, with how long ago each reported, what
their jobs are doing, and how many finished runs they have not read. Click a
name for that person's dashboard; tick several and open them together, and
every job card then says whose it is.

Each user and each project carries a colour, shown on its card and on every job
tag. One is assigned the first time either appears and can be changed from the
swatch on its card. The eight on offer come from a documented categorical
palette, checked against this dashboard's dark surface: every one clears 3:1,
and neighbouring pairs stay far enough apart for a colour-blind reader. They are
handed out in palette order rather than at random, because any two tags can end
up side by side and only the first few slots survive that test -- which is also
why every tag carries its name as text. Colour is what makes a tag findable at a
glance; the word is what says which one it is.

**Projects** is what was being run. A project collects jobs across users and
across submission batches, because one experiment is usually several people's
grids submitted hours apart -- which the user cut can only ever show
separately. Open one and it reads exactly like a user's dashboard, still
grouped by submission batch, with each card naming its owner.


- Running, pending, failed and completed jobs, newest job id first, grouped by
  the batch they were submitted in -- everything landing within fifteen minutes
  of the newest job in a group. A grid goes in as a burst of sbatch calls, so a
  group is usually one experiment plus whatever went in around it, and its
  heading says how the whole thing is getting on: "9 jobs · 9 failed" without
  reading nine cards.
- Per job: state, requested resources (GPU, memory, CPUs, partition), wall time
  used against the limit, and time remaining.
- A run name too long for its card drifts sideways at a steady 35 px/s, pausing
  at each end, so the part that names the experiment is readable rather than
  eaten by an ellipsis. Only names that actually overflow move, and none do for
  anyone who has asked for reduced motion -- the hover title carries them
  instead.
- `Reason` on pending jobs — which is how a stalled `afterok` chain from
  `slurm/submit_graph_grid.sh` becomes visible: downstream jobs sit in `PENDING`
  with `Reason=Dependency` after an early failure.
- The complete log of each job, opening at the end and tailing live, with an
  errors-only filter and a full download.
- A time window (1h / 6h / 24h / 7d / 30d) selecting jobs that were **running or
  stopped inside that window**.
- A run that finished since you last opened it glows in its outcome's colour
  until you read it. Opening it clears the glow; "Mark N read" clears every
  glowing run currently on screen at once.
- The dashboard's own agent job is kept out of the list -- it is infrastructure,
  not an experiment. Its health is the banner, and the job id there opens it.
  Retired agents are recognised by job name, since successive agents share one:
  `is_agent` alone means "this job is me", which no past agent can report.
- The latest `sres` GPU availability as a searchable table, each node scored on
  how free it is and tinted red to green to match.

## Security

Read access is **not** authenticated: anyone with the URL can see your job
names, file paths and logs. Only ingest is gated, by a shared token, so an
outsider cannot poison the job list. Deploy it on an unguessable URL, and if you
later want the read side gated too, that is a small change to `server/app/main.py`.

Nothing the dashboard does can affect the cluster. There is no `scancel` and no
resubmission of your jobs. Nothing can be deleted from the dashboard either:
the only thing a click changes is whether a finished run is still glowing.

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
agent re-ships them from the beginning on the next poll. It asks for the job
history the same way -- a server holding nothing but the agent's own job says
so, and gets a full `sacct` sweep on the next poll rather than leaving the
dashboard empty until the scheduled one comes round five minutes later.

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

Three places are checked for it, in order: `$DASHBOARD_ENV_FILE`, then
`dashboard/.dashboard-env` beside the checkout, then the path above. The middle
one keeps a deployment self-contained -- useful when a checkout may not write
outside its own tree -- and because the agent resubmits its successor from the
same directory, it is found again on every handover without depending on the
environment surviving `sbatch`. Set `DASHBOARD_STATE_DIR` in that file to move
the agent's offsets and lock alongside it.

### More than one person

Each person runs the same agent from their own account, against the same
`DASHBOARD_URL` and token. The agent reports whoever `$USER` says it is -- so
there is nothing to configure -- and the server keeps one heartbeat per user
rather than one in total. Nobody's jobs, logs or read marks are separated by
anything but that name, and the shared ingest token is what every agent
presents, so this shares a dashboard between people who already share a
cluster account boundary, not between strangers.

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

## Filing jobs into a project

By hand: create one from the entry screen, then tick jobs in the list and pick
a project from the bar that appears. The checkbox on a batch heading takes the
whole batch at once, which is usually what a project is made of, and the bar
can create a project and file into it in the same gesture. A single job can
also be moved from its detail pane.

Automatically, which is the point: whatever submits a grid already knows the
ids `sbatch` just returned, so handing them over is one more call:

```bash
DASH=https://your-app.example.com

# Idempotent by id, so a submitter can call it on every run without checking.
curl -sX POST "$DASH/api/projects" \
     -H 'Content-Type: application/json' \
     -d '{"name": "Gate ablation r10-30"}'
# -> {"id": "gate-ablation-r10-30", "name": "Gate ablation r10-30", ...}

# Then the whole grid in one call.
curl -sX POST "$DASH/api/projects/gate-ablation-r10-30/jobs" \
     -H 'Content-Type: application/json' \
     -d '{"job_ids": ["20824155", "20824156", "20824157"]}'
# -> {"project_id": "gate-ablation-r10-30", "assigned": 3}
```

Ids are slugs of the name so they read well in an address bar and in a
submitter's own logs. Unknown job ids are skipped rather than failing the
batch, so handing over a whole grid does not break because one id has since
been pruned; assigning to a project that does not exist is a 404, because that
is a mistake worth hearing about. `DELETE` on the same path takes jobs back
out, and deleting a project keeps its jobs -- they simply belong to nothing.

A job is in at most one project, and `project_id` is never touched by ingest:
it is not something SLURM knows, so the agent would otherwise wipe it on every
poll.

Like the read marks, these writes are not authenticated. They record how
somebody wants their own runs arranged and cannot reach the cluster; only
ingest is gated, so the job list itself still cannot be poisoned.

## How the availability score works

Every `sres` row is scored in [0, 1] and tinted from red to green:

```
score = gpu^0.6 * mem^0.2 * cpu^0.2
```

The GPU term is measured against **what one user may hold**, not against the
node: `min(free, 5) / 5`. Five is this cluster's cap -- the `gpu` partition runs
QOS `gpu-part`, whose `MaxTRESPU` is `gres/gpu=5`, which is what a job means
when it pends with `Reason=QOSMaxGRESPerUser`. It does not show on a user's own
QOS, so `sacctmgr show qos where name=<yours>` will not find it; look at the
partition's.

Scoring against the node's total asked the wrong question. Twenty free GPUs and
five free GPUs are the same offer to someone who may hold five, and dividing by
the total made a node that could take a whole allocation look worse than one
that could not. Memory and CPUs are still fractions of the node, because
nothing in `sres` says how much of either a given job will ask for. Resources `sres` does not report
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

If the panel reads **never reported**, `sres` could not be run where the agent
is. It is not a real executable, so only a shell can run it: the agent tries an
interactive login shell first -- an alias resolves in no other kind, and a
function defined in `~/.bashrc` is invisible to `bash -lc`, which sources only
`~/.bash_profile` -- then the plainer forms. When every attempt fails it stores
what went wrong, so the panel shows the reason rather than disappearing. Run
`type sres` on the login node and on a compute node: if it exists only on the
login node, the agent cannot reach it from wherever SLURM placed it.

The panel also keeps the `GPU UTILIZATION` block above the node table, as a
tinted chip per GPU type. That block is the only place the output names a
*type*: the node table counts GPUs per node without saying whether they are
6000pro or 6000, which is the choice being made before a submission.

Nothing is hard-coded to a column layout, because `sres` is site-local with no
stable documented format. The node table is found by trying every line as a
header and keeping the one whose following lines split to the same width and
carry counts -- BGU's output puts a banner and the utilisation block above it,
so it never starts at line one. Columns are separated on runs of two or more
spaces, because a single space falls *inside* a cell (`3 / 3`) and inside a
header (`MEM [GB]`); plain whitespace is tried as a fallback. Any column whose
cells read as `free/total` is taken for a resource, named by its own header or
by the header of the column naming it (`GPU  FREE`). When nothing can be
identified the panel falls back to the raw output, which the "Raw" button also
shows on demand.

## How the log transfer works

Logs from these runs reach hundreds of megabytes, mostly tqdm progress lines, so
they are never re-uploaded whole. The agent tracks a byte offset per job and
ships only what is new, capped at 512 KB per job per poll; the server appends it
to a plain file. The result is the complete log on the server, at the cost of
the bytes actually written.

The server, not the agent, is the authority on how much of each log it holds. It
answers every poll with the true offsets, so the two re-align by themselves
after a lost response, a wiped disk, or a job that rewrote its log in place.
