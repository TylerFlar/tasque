# tasque

Single-user task-orchestration daemon. Nine life-area "buckets" (health,
relationships, education, career, finance, creative, home, personal,
recreation) each have a coach agent that watches incoming notes and queues
worker jobs. A strategist sits above the coaches for cross-bucket work.
Discord is the conversational surface; the CLI is the read interface.

## Architecture in one paragraph

tasque is the orchestration layer around your host's Claude CLI. Your
host MCPs (calendar, browser, filesystem, course access, etc.) plug in
via `~/.claude.json`; the daemon's LLM calls go through `tasque proxy`,
which wraps `claude --print` as an OpenAI-compat endpoint, so every host
MCP is inherited transparently. tasque also ships **its own MCP**
(`tasque mcp`) you register the same way — that gives every coach,
worker, and strategist run a live tool catalog for the daemon's own
state (notes, queued jobs, chain templates, chain runs, signals). Memory
is SQLite via SQLAlchemy. Job scheduling is APScheduler. Multi-step
workflows ("chains") are LangGraph with the SQLite checkpointer. One
process, one venv, no Docker.

## Install

```bash
uv sync
cp .env.example .env   # fill in TASQUE_DISCORD_TOKEN and channel ids
```

`.env` is auto-loaded into `os.environ` (pydantic-settings populates the
typed `Settings` object for paths and decay windows; the rest of the
codebase reads its env vars directly). See [.env.example](.env.example)
for every knob (paths, proxy, model selection, coach dedup, Discord
ids).

## Daily use

```bash
# Start the daemon. One asyncio loop running:
#   - the Discord bot
#   - the APScheduler-driven job scheduler
#   - the coach trigger drainer
#   - the chain status watcher (live-edits per-run status embeds)
#   - the worker run watcher (posts worker output to the jobs channel)
# The bot's on_ready also installs the ops-panel watcher (the live
# /status embed in the ops channel).
uv run tasque serve

# Start the LLM proxy in another shell — the daemon expects it on :3456.
uv run tasque proxy
```

Both are long-lived. Stop with Ctrl-C; shutdown is cooperative.

For an at-a-glance state read without opening Discord:

```bash
uv run tasque status            # JSON, pipe into jq
uv run tasque status --text     # human-readable
```

## CLI overview

```
tasque memory   import | export | wipe | prune | stats
tasque proxy                                  # OpenAI-compat wrapper around claude --print
tasque mcp                                    # tasque stdio MCP server (register in ~/.claude.json)
tasque coach    wake <bucket>
tasque jobs     queue | list | stop | tick
tasque dlq      list | show | retry | resolve
tasque chain    reload | export | queue | list | show | templates | pause | resume | stop | delete
tasque serve                                  # the daemon
tasque status   [--text]                      # snapshot of jobs, chains, DLQ, scheduler
```

Run any subcommand with `--help` for the full option list.

## Memory: import, wipe, reimport

```bash
# import a JSONL file (one {"type": "<EntityName>", ...} object per line)
uv run tasque memory import path/to/data.jsonl

# import a directory of markdown notes — files under a bucket-named subdir
# (health, relationships, education, career, finance, creative, home,
# personal, recreation) inherit that bucket
uv run tasque memory import path/to/notes/

# wipe the database file (irreversible)
uv run tasque memory wipe --yes

# reimport from the JSONL you exported earlier
uv run tasque memory export out.jsonl   # before the wipe
uv run tasque memory wipe --yes
uv run tasque memory import out.jsonl

# row counts per entity type
uv run tasque memory stats

# archive ephemeral / superseded / expired rows (and optionally hard-delete
# long-archived ones). Defaults read from settings; --dry-run reports counts
# without changing anything.
uv run tasque memory prune --dry-run
uv run tasque memory prune --hard-delete-days 90
```

## Schema evolution

Column additions happen automatically at startup via `_ensure_schema()` —
new columns get ALTER-ADDed onto existing tables, so adding a field to an
entity costs nothing. Type changes and column removals go through
`tasque memory export` → `tasque memory wipe --yes` → `tasque memory
import`. There is no Alembic-style migration framework; the
[migration/](migration/) directory holds the one-off scripts and JSONL
artefacts from past schema reshapes, kept for reference.

## LLM proxy

`tasque proxy` wraps the host's `claude --print` CLI as an OpenAI-compat
chat-completions endpoint on `http://localhost:3456/v1`. The daemon's
ChatOpenAI clients point at this URL, so every MCP configured in your
host `~/.claude.json` (calendar, browser, filesystem, …) is inherited
without re-implementing MCP plumbing.

```bash
uv run tasque proxy                 # binds 127.0.0.1:3456
uv run tasque proxy --port 3500     # custom port
TASQUE_PROXY_MAX_CONCURRENT=8 uv run tasque proxy
TASQUE_PROXY_TIMEOUT=120 uv run tasque proxy   # per-request wall-clock cap
```

Knobs:

- `TASQUE_PROXY_MAX_CONCURRENT` (default 4) — semaphore cap on in-flight
  `claude --print` invocations. Higher than 1 so nested LLM calls through
  MCPs that route back through the proxy don't deadlock.
- `TASQUE_PROXY_TIMEOUT` — optional float seconds applied to
  `subprocess.communicate(timeout=…)`. Unset means no timeout; trust outer
  guards (the daemon sets per-step timeouts).
- `TASQUE_PROXY_LOG_DIR` (default `data/proxy-logs`) — per-request raw
  stream-json transcripts, named `<request_id>.jsonl`. Files older than
  7 days are pruned at startup.

Endpoints: `POST /v1/chat/completions`, `GET /healthz` (200 when
`claude --version` succeeds, 503 otherwise; cached 30 s),
`GET /status` (in-flight count, capacity, totals, last error).

## tasque MCP

`tasque mcp` is a stdio MCP server that exposes the daemon's own state
(notes, queued jobs, chain templates, chain runs, signals) as live
tools. Registering it in `~/.claude.json` puts those tools in front of
every LLM call routed through the proxy — so the reactive bucket coach,
the chain worker, the planner, and the strategist can all fire chains,
queue jobs, edit templates, and write notes mid-turn. Without it,
those agents have to encode every action in trailing JSON, which
collapses for any "do this *now* and observe the result before
continuing" need.

Register it once:

```jsonc
// ~/.claude.json
{
  "mcpServers": {
    "tasque": {
      "command": "uv",
      "args": ["run", "--directory", "/abs/path/to/tasque", "tasque", "mcp"]
    }
  }
}
```

(Or omit `uv` and use a virtualenv-relative path to the `tasque`
console script — whichever spawns the right venv.)

Tools exposed (full schemas are emitted on connect — your MCP host
will list them):

- **Notes** — `note_create`, `note_get`, `note_list`, `note_search`,
  `note_search_any`, `note_search_fts`, `note_archive`.
- **Queued jobs** — `job_create`, `job_get`, `job_update`, `job_cancel`,
  `job_list`. Both one-shot and recurring (5-field cron, alias DOW).
- **Chain templates** — `chain_template_create`, `chain_template_get`,
  `chain_template_list`, `chain_template_update`, `chain_template_delete`.
- **Chain runs** — `chain_fire_template`, `chain_queue_adhoc`,
  `chain_run_get`, `chain_run_list`, `chain_run_pause`,
  `chain_run_resume`, `chain_run_stop`.
- **Signals** — `signal_create`, `signal_list`, `signal_archive`.

Bucket scoping is by explicit argument: every write that touches a
bucket takes a `bucket` parameter, validated against the nine canonical
names. The agent's system prompt tells it which bucket it is; the LLM
passes that value through. Mutations return `{"ok": true, ...}` on
success and `{"ok": false, "error": "…"}` on validation/runtime
failure.

The MCP and the daemon share the same SQLite DB (`<DATA_DIR>/tasque.db`)
under WAL — concurrent reads from the daemon and writes from an MCP
session are fine. There is no separate auth: the MCP server runs on
your host, scoped to your account, talking to your local DB.

### MCP smoke test

With the MCP registered:

```bash
# In Claude Code or any MCP-aware host, you should see "tasque" listed
# alongside your other MCPs and 26 tools available. From inside a model
# turn:
#   chain_template_list()                          → JSON array of templates
#   note_create(content="hello", bucket="personal") → {"ok": true, "id": "..."}
```

### Proxy smoke test

With the Claude CLI installed and authenticated:

```bash
uv run tasque proxy &

curl -s -X POST localhost:3456/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"claude-haiku-4-5","messages":[{"role":"user","content":"reply with the single word: OK"}]}'
# → JSON body whose choices[0].message.content contains "OK"

curl -s localhost:3456/healthz   # → {"status":"ok"}
```

## Model selection

Two paths, depending on agent kind:

**Coach + strategist** (fixed-role) resolution priority (highest first):

1. `TASQUE_MODEL_<KIND>` — per-agent override (`COACH`, `STRATEGIST`).
2. `TASQUE_MODEL_<TIER>` — tier override (`OPUS`, `SONNET`, `HAIKU`).
3. Built-in defaults — both default to the opus tier.

**Worker + chain planner**: tier is chosen per row (per `QueuedJob.tier`
and per chain plan node / chain spec `planner_tier`). The kind-level
env overrides do **not** apply here — only the tier override and the
built-in defaults do.

See [src/tasque/llm/factory.py](src/tasque/llm/factory.py) for the
current default model ids.

## Coaches

Per-bucket system prompts live in [coach_prompts/](coach_prompts/) — one
markdown file per bucket. The shared scaffolding (tool list, output
format, time block) lives in [prompts/coach_scaffold.md](prompts/coach_scaffold.md)
and is concatenated at request time, so per-bucket files only carry the
mindset.

The coach is fired exclusively through a single SQLite-backed trigger
queue. Worker completion, Discord replies, and scheduled wakes all
`enqueue(bucket, reason, dedup_key)`; the drainer claims rows serially
per bucket and runs the coach LangGraph. Dedup is two-phase against
`(bucket, dedup_key)`:

- **Pending**: any unclaimed pending row collapses the new enqueue
  unconditionally — no time bound.
- **Post-claim**: a row that was claimed within the dedup window also
  collapses. Default is 300 seconds (5 minutes), overridable via
  `TASQUE_COACH_DEDUP_SECONDS`.

`tasque coach wake` passes `dedup_key=None`, which always enqueues —
that's the explicit-wake path.

```bash
# Manual wake — bypasses dedup (uses dedup_key=None).
uv run tasque coach wake health --reason "morning check-in"
```

## Strategist

The strategist is the cross-bucket layer above the coaches. Two
surfaces:

- **Decomposition** — converse with the strategist on its Discord
  thread. Describe a long-horizon Aim and it breaks it into
  per-bucket Aims and Signals (the conversational reply runtime in
  [src/tasque/reply/strategist.py](src/tasque/reply/strategist.py)).
- **Monitoring** — a scheduled cross-bucket survey runs the
  LangGraph in [src/tasque/strategist/graph.py](src/tasque/strategist/graph.py).
  It gathers recent Notes, Signals, and failures across all buckets,
  posts a markdown summary, and emits new Aims/Signals (or flips stale
  Aim statuses). Wire it in via a chain step whose `directive` is the
  sentinel `[strategist:monitor]`; the worker dispatcher routes that
  to the monitoring graph instead of an LLM call.

The strategist uses the opus tier by default, same resolution rules as
coaches (`TASQUE_MODEL_STRATEGIST` → `TASQUE_MODEL_OPUS` → built-in).

## Jobs

`QueuedJob` is the execution unit for both standalone tasks and chain
steps. The scheduler is APScheduler-backed and serial: one job at a time,
heartbeat-based liveness, recurring jobs via cron.

```bash
uv run tasque jobs queue "summarise unread email" --bucket personal --fire-at now
uv run tasque jobs queue "weekly review" --bucket career --cron "0 9 * * MON"
uv run tasque jobs list --status pending
uv run tasque jobs stop <job-id>
uv run tasque jobs tick                 # run one tick synchronously (for testing)
```

Cron expressions use the standard 5-field form. Pure-numeric day-of-week
(e.g. `1-5`) is rejected at insert time — use the alias form (`MON-FRI`)
to avoid the off-by-one trap.

## Dead-letter queue

When a worker run fails, the scheduler writes a `FailedJob` row and flips
the `QueuedJob` to `failed`. The DLQ surfaces in Discord and via the CLI:

```bash
uv run tasque dlq list
uv run tasque dlq show <failed-job-id>
uv run tasque dlq retry <failed-job-id>     # re-fires (chain steps go through chain hook)
uv run tasque dlq resolve <failed-job-id>   # manual closeout
```

## Chains

A chain is a multi-step LangGraph plan: workers, approvals, ask-user
prompts, and planner mutations. `ChainTemplate` rows hold the canonical
plan (`plan_json`); `ChainRun` rows are the discoverable handles for live
executions, with the actual step state in the LangGraph SQLite
checkpointer.

Templates are authored as YAML in `chains/templates/` (the directory
is created on first reload) and loaded with `tasque chain reload`,
which scans the directory and upserts each template into the DB. Edit
the YAML, re-run `reload`. To move state the other way (e.g. inspect
a row whose YAML you've lost), `tasque chain export <name>` writes the
row back out to YAML.

```bash
uv run tasque chain reload                            # upsert chains/templates/*.yaml
uv run tasque chain reload --path other/templates     # custom directory
uv run tasque chain export <name> [path]              # template row -> YAML

uv run tasque chain queue <template-name>             # one-shot from template
uv run tasque chain queue path/to/spec.json           # ad-hoc one-shot
uv run tasque chain templates --enabled-only
uv run tasque chain list --status running
uv run tasque chain show <chain-id>                   # render plan tree
uv run tasque chain pause  <chain-id>
uv run tasque chain resume <chain-id>
uv run tasque chain stop   <chain-id>
uv run tasque chain delete <template-name>
```

Mirror columns on `ChainTemplate` (`recurrence`, `bucket`) must match the
values inside `plan_json`. Every write goes through `validate_spec`,
which rejects mismatches at the seam.

## Discord

One nextcord bot. Five channels, each with a single distinct purpose
— do **not** point two of them at the same channel unless you want
the streams interleaved:

- **Coach channel** — parent for the per-bucket coach threads.
  Reserved for user ↔ coach conversation.
- **Jobs channel** — worker-run embeds; per-chain threads anchor here.
  Per-chain threads host host-approval and ask-user prompts as embeds
  with buttons.
- **DLQ channel** — failed-job entries with a Retry button.
- **Ops channel** — one live-edited ops embed (jobs / chains / DLQ
  snapshot), refreshed every 30s. Pin the message manually for
  visibility. The same data is available offline via `tasque status`.
- **Chains channel** — one live-edited status embed per `ChainRun`,
  refreshed as the run progresses (node start, fan-out, completion).

There is also a `/status` slash command that returns the ops snapshot
to whoever invoked it.

### Setup

Single-user bot, single guild. The bot creates and posts in threads
underneath the coach channel and the jobs channel; per-bucket threads,
per-chain threads, and any anchor messages are spawned automatically
on first use.

**1. Create the application + bot.**

1. Go to <https://discord.com/developers/applications> and click *New
   Application*. Name it whatever you like.
2. Open the **Bot** tab. Copy the token (or *Reset Token* the first
   time) — this is `TASQUE_DISCORD_TOKEN`. Treat it like a password.
3. On the same tab, under *Privileged Gateway Intents*, enable
   **Message Content Intent**. The reply runtime reads message bodies,
   so the bot will silently see empty content without it. Server
   Members and Presence intents are not needed.
4. Under *Bot Permissions* (or *Default Install Settings* on newer
   portals), the bot needs: View Channels, Send Messages, Send Messages
   in Threads, Create Public Threads, Read Message History, Embed
   Links, Attach Files, Manage Messages (for editing the chain
   approval embeds when buttons resolve).

**2. Invite the bot to your server.**

Open the **OAuth2 → URL Generator**, tick the `bot` scope, then tick
the permissions listed above (the integer permission flag is
`377957403648` if you'd rather paste it directly into the URL). Open
the generated URL, pick your server, authorize. You need *Manage
Server* on the target guild to add the bot — fine for a personal
guild, since you own it.

**3. Create the five channels.**

In your server, make five regular text channels (e.g. `#tasque-coach`,
`#tasque-jobs`, `#tasque-dlq`, `#tasque-ops`, `#tasque-chains`). Daemon
startup fails fast if any of the five ids is unset or unparseable — there
are no fallbacks.

Right-click each channel → *Copy Channel ID* (enable *Settings →
Advanced → Developer Mode* first). The bot creates the per-bucket
threads (under the coach channel) and per-chain threads (under the
jobs channel) itself on first start; you do not need to pre-create
those threads.

**4. Wire `.env`.**

```bash
TASQUE_DISCORD_TOKEN=...                          # from step 1
TASQUE_DISCORD_COACH_CHANNEL_ID=...               # required
TASQUE_DISCORD_JOBS_CHANNEL_ID=...                # required
TASQUE_DISCORD_DLQ_CHANNEL_ID=...                 # required
TASQUE_DISCORD_OPS_CHANNEL_ID=...                 # required
TASQUE_DISCORD_CHAINS_CHANNEL_ID=...              # required
# TASQUE_DISCORD_GUILD_ID=...                     # optional; instant /status registration in dev
```

**5. Start the daemon.**

```bash
uv run tasque proxy &       # other shell, or any process supervisor
uv run tasque serve
```

On first connect the bot logs `discord.bot.ready`, then creates one
thread per bucket under the coach channel and posts the live ops embed
under the ops channel (pin it manually if you want it sticky). Thread
ids are persisted in `<DATA_DIR>/discord_threads.json` (override path
via `TASQUE_DISCORD_THREAD_REGISTRY`), so subsequent restarts reuse
them instead of spawning duplicates. If you delete a thread on Discord,
the bot recreates it on next startup. The ops-embed message id is
cached the same way; deleting the message causes the watcher to repost
on the next refresh tick.

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run pyright
```

Tests run against a temp SQLite DB and a stubbed proxy — no live Claude
CLI or Discord token required.
