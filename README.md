# Frankenbote

A personal weekly news digest for Franconia, Bavaria, and Germany — generated automatically by an AI pipeline that fetches RSS feeds, curates articles with Claude, and publishes a static HTML newsletter to a web server.

---

## How it works

The pipeline runs in sequential stages:

1. **Fetch** — pulls RSS/Atom feeds from configured news sources
2. **Filter** — drops articles outside the weekly time window and applies keyword/category rules
3. **Curate** — sends candidates to Claude, which assigns sections and priority scores (P1–P4), dropping low-relevance items
4. **Select** — assembles the final edition respecting priority quotas and target size
5. **Summarize** — calls Claude again to write a short AI summary for each article
6. **Render** — generates static HTML from Jinja2 templates into `output/`
7. **Publish** — uploads `output/` to a remote web server via SFTP

---

## Setup

### Prerequisites

- Python 3.14+
- An [Anthropic API key](https://console.anthropic.com/)
- (Optional) Docker, for running in a container
- (Optional) An SFTP-accessible web server for publishing

### Local installation

```bash
# Clone the repo
git clone <repo-url>
cd frankenbote

# Create and activate a virtual environment
python3 -m venv frankenbote-env
source frankenbote-env/bin/activate

# Install the package and its dependencies
pip install -e .
```

### Configuration

Copy the example environment file and fill in the required values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | API key from console.anthropic.com |
| `FRANKENBOTE_ENV` | No | `development` (default) or `production` |
| `LOG_LEVEL` | No | `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`) |
| `SFTP_HOST` | Publishing only | Hostname of your web server |
| `SFTP_USERNAME` | Publishing only | SFTP login username |
| `SFTP_PRIVATE_KEY_PATH` | Publishing only | Path to your SSH private key |
| `SFTP_REMOTE_ROOT` | Publishing only | Remote directory to deploy into |
| `SFTP_PORT` | No | SFTP port (default: `22`) |
| `SFTP_PRIVATE_KEY_PASSPHRASE` | No | Passphrase if key is protected |

The YAML files in `config/` control what gets fetched and how articles are categorised:

- `config/sources.yaml` — RSS feed list; toggle sources on/off with `enabled: true/false`
- `config/filter.yaml` — time window and keyword filtering rules
- `config/sections.yaml` — section definitions and AI curation model settings

---

## Running locally

### Verify setup

```bash
frankenbote hello
```

Prints the detected environment, Python version, and whether the API key is present.

### Run the full pipeline

```bash
frankenbote pipeline
```

This is the main command. It runs all stages end-to-end and, if SFTP is configured, publishes the result. Open `output/index.html` in a browser to preview locally.

### Run with Docker

```bash
# Build the image
docker compose build

# Run the full pipeline
docker compose run --rm app pipeline

# Run any other subcommand
docker compose run --rm app hello
```

The `docker-compose.yml` mounts `src/`, `config/`, `templates/`, `assets/`, `data/`, and `output/` from the host, so local edits are picked up immediately without rebuilding.

---

## CLI Reference

All commands are available as subcommands of `frankenbote`. Run `frankenbote --help` or `frankenbote <command> --help` for full option details.

### `hello`

```
frankenbote hello
```

Verify that the tool is installed and the environment is configured correctly.

### `fetch`

```
frankenbote fetch [--config config/sources.yaml]
```

Fetch all enabled RSS sources and print a per-source article count. Useful for checking that feeds are reachable before running the full pipeline.

### `pipeline`

```
frankenbote pipeline [--sources config/sources.yaml]
                     [--filter-config config/filter.yaml]
                     [--sections-config config/sections.yaml]
                     [--size 25]
                     [--no-curate]
```

Run all stages in sequence. `--size` sets the target number of articles in the final edition (5–100). Pass `--no-curate` to stop after filtering, which skips all LLM calls — useful for development.

### `curate`

```
frankenbote curate --candidates-date YYYY-MM-DD
```

Re-run the AI curation step on a previously saved candidates file without refetching. The date must match a file in `data/`.

### `select`

```
frankenbote select --curated-date YYYY-MM-DD [--size 25]
```

Re-run article selection on a previously curated dataset. Useful for experimenting with different edition sizes without repeating the LLM call.

### `summarize`

```
frankenbote summarize --edition-date YYYY-MM-DD
```

Re-run AI summarization on a previously saved edition JSON.

### `render`

```
frankenbote render
```

Re-render all saved editions to HTML in `output/`. Run this after changing templates or CSS.

### `publish`

```
frankenbote publish
```

Upload the `output/` directory to the configured SFTP server. Requires the `SFTP_*` environment variables to be set.

---

## Project structure

```
frankenbote/
├── config/           # YAML configuration (sources, filter rules, sections)
├── src/frankenbote/  # Application source code
│   ├── cli.py        # Click CLI entry point
│   ├── fetcher.py    # Async RSS fetching
│   ├── filter.py     # Time-window and keyword filtering
│   ├── curator.py    # AI curation via Claude
│   ├── selector.py   # Priority-based article selection
│   ├── summarizer.py # AI summarization via Claude
│   ├── renderer.py   # Jinja2 HTML rendering
│   └── publisher.py  # SFTP publishing
├── templates/        # Jinja2 HTML templates
├── assets/           # Static CSS and icons
├── data/             # Intermediate pipeline data (candidates, curated JSON)
├── output/           # Final rendered HTML (git-ignored)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## License

See [LICENSE](LICENSE).
