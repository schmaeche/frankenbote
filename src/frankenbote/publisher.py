"""Publisher — uploads the rendered output/ tree to a remote host via SFTP.

Provider-agnostic: works with any hosting that exposes SFTP with SSH-key
authentication. Tested against shared webhosting (single-user SFTP into a
chrooted filesystem) but should also work for VPSes and similar setups.

Reads:
  - output/                       (everything the renderer produced)
  - .env                          (SFTP_* credentials)

Writes (on the remote server):
  - <remote_dir>/index.html
  - <remote_dir>/editions/YYYY-MM-DD.html
  - <remote_dir>/assets/style.css
  - <remote_dir>/assets/frankenrechen.svg

Deletes (defensively, only files matching the editions pattern):
  - <remote_dir>/editions/*.html files not in the local set

NEVER touches anything outside <remote_dir>. NEVER deletes index.html
or anything in assets/. Defensive design — this is a destructive operation
running over the network against a real production environment.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

import click
import paramiko


# Strict: edition HTML filename must look like 'YYYY-MM-DD.html'.
# Used by the prune step to avoid touching anything that doesn't match.
_EDITION_FILENAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.html$")


@dataclass
class PublisherConfig:
    """SFTP connection details and target paths."""

    host: str
    username: str
    private_key_path: Path
    private_key_passphrase: str | None  # None if key has no passphrase
    remote_dir: str
    port: int = 22
    local_output_dir: Path = Path("output")


def load_publisher_config_from_env() -> PublisherConfig:
    """Build a PublisherConfig from environment variables.

    Required:
      SFTP_HOST                    e.g. example.com
      SFTP_USERNAME                e.g. webuser
      SFTP_PRIVATE_KEY_PATH        path to the private key file
      SFTP_REMOTE_ROOT             remote target directory. May be absolute
                                   (/var/www/html) or relative to the SFTP
                                   session's working directory (public_html).

    Optional:
      SFTP_PRIVATE_KEY_PASSPHRASE  if the private key is passphrase-protected
      SFTP_PORT                    defaults to 22
    """
    required = {
        "SFTP_HOST": os.environ.get("SFTP_HOST"),
        "SFTP_USERNAME": os.environ.get("SFTP_USERNAME"),
        "SFTP_PRIVATE_KEY_PATH": os.environ.get("SFTP_PRIVATE_KEY_PATH"),
        "SFTP_REMOTE_ROOT": os.environ.get("SFTP_REMOTE_ROOT"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return PublisherConfig(
        host=required["SFTP_HOST"],
        username=required["SFTP_USERNAME"],
        private_key_path=Path(required["SFTP_PRIVATE_KEY_PATH"]).expanduser(),
        private_key_passphrase=os.environ.get("SFTP_PRIVATE_KEY_PASSPHRASE") or None,
        remote_dir=required["SFTP_REMOTE_ROOT"],
        port=int(os.environ.get("SFTP_PORT", "22")),
    )


# ---------- Public API ----------


def publish(config: PublisherConfig | None = None) -> dict[str, int]:
    """Upload local output/ to the configured SFTP target.

    Returns a stats dict for CLI reporting.
    """
    config = config or load_publisher_config_from_env()

    if not config.local_output_dir.exists():
        raise RuntimeError(
            f"Local output dir not found: {config.local_output_dir}. "
            "Run `frankenbote render` first."
        )
    if not config.private_key_path.exists():
        raise RuntimeError(
            f"SSH private key not found: {config.private_key_path}"
        )
    
    click.echo(f"Key path exists: {config.private_key_path}")

    stats = {"uploaded": 0, "pruned": 0}

    with _connect(config) as sftp:
        # Make sure the remote layout exists.
        _ensure_remote_dir(sftp, config.remote_dir)
        _ensure_remote_dir(sftp, f"{config.remote_dir}/editions")
        _ensure_remote_dir(sftp, f"{config.remote_dir}/assets")

        local_editions_dir = config.local_output_dir / "editions"
        local_assets_dir = config.local_output_dir / "assets"
        local_index = config.local_output_dir / "index.html"

        # Assets first, then editions, then the index that links to them.
        if local_assets_dir.is_dir():
            for asset in sorted(local_assets_dir.iterdir()):
                if asset.is_file():
                    _upload_file(sftp, asset, f"{config.remote_dir}/assets/{asset.name}")
                    stats["uploaded"] += 1

        local_edition_filenames: set[str] = set()
        if local_editions_dir.is_dir():
            for edition in sorted(local_editions_dir.iterdir()):
                if edition.is_file() and _EDITION_FILENAME_RE.match(edition.name):
                    local_edition_filenames.add(edition.name)
                    _upload_file(sftp, edition, f"{config.remote_dir}/editions/{edition.name}")
                    stats["uploaded"] += 1

        if local_index.is_file():
            _upload_file(sftp, local_index, f"{config.remote_dir}/index.html")
            stats["uploaded"] += 1

        # Prune remote editions not in our local set.
        # CRITICAL: only touch files that match the strict edition pattern.
        stats["pruned"] = _prune_remote_editions(
            sftp,
            f"{config.remote_dir}/editions",
            keep=local_edition_filenames,
        )

    return stats


# ---------- Internals ----------


def _connect(config: PublisherConfig):
    """Open an SFTP connection. Caller must use as a context manager."""
    transport = paramiko.Transport((config.host, config.port))
    pkey = paramiko.Ed25519Key.from_private_key_file(
        str(config.private_key_path),
        password=config.private_key_passphrase,
    )
    transport.connect(username=config.username, pkey=pkey)
    sftp = paramiko.SFTPClient.from_transport(transport)
    if sftp is None:
        transport.close()
        raise RuntimeError("Failed to open SFTP channel")
    return _SFTPSession(sftp, transport)


class _SFTPSession:
    """Context manager that closes both SFTP and Transport cleanly."""

    def __init__(self, sftp: paramiko.SFTPClient, transport: paramiko.Transport):
        self._sftp = sftp
        self._transport = transport

    def __enter__(self) -> paramiko.SFTPClient:
        return self._sftp

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._sftp.close()
        finally:
            self._transport.close()


def _ensure_remote_dir(sftp: paramiko.SFTPClient, path: str) -> None:
    """Create the remote dir if it doesn't exist. Idempotent.

    Only creates the leaf — parent directories must already exist.
    """
    try:
        attr = sftp.stat(path)
        if not stat.S_ISDIR(attr.st_mode):
            raise RuntimeError(f"Remote path {path} exists but is not a directory")
    except FileNotFoundError:
        sftp.mkdir(path)


def _upload_file(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    """Upload a single file. Overwrites without prompting."""
    sftp.put(str(local), remote)


def _prune_remote_editions(
    sftp: paramiko.SFTPClient,
    remote_editions_dir: str,
    keep: set[str],
) -> int:
    """Delete edition files in the remote dir that aren't in `keep`.

    Defensive: only touches files whose names match the strict edition
    pattern (YYYY-MM-DD.html). Anything else is left alone.
    """
    pruned = 0
    try:
        entries = sftp.listdir(remote_editions_dir)
    except FileNotFoundError:
        return 0

    for name in entries:
        if not _EDITION_FILENAME_RE.match(name):
            # Not an edition file — never touch.
            continue
        if name in keep:
            continue
        sftp.remove(f"{remote_editions_dir}/{name}")
        pruned += 1
    return pruned