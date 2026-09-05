#!/usr/bin/env python3
"""Discover distinct SSH users whose aliases resolve to one cluster endpoint."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


PATTERN_CHARS = frozenset("*?!")


def _tokens(line: str) -> list[str]:
    try:
        return shlex.split(line, comments=True, posix=True)
    except ValueError as exc:
        raise ValueError(f"invalid SSH config line: {line.rstrip()}: {exc}") from exc


def _literal_alias(token: str) -> bool:
    return bool(token) and not token.startswith("!") and not any(
        char in token for char in PATTERN_CHARS
    )


def collect_aliases(config: Path) -> list[str]:
    """Collect literal Host aliases, following user-config Include directives."""

    config = config.expanduser().resolve()
    include_base = config.parent
    aliases: set[str] = set()
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved in visited or not resolved.is_file():
            return
        visited.add(resolved)
        for line in resolved.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = _tokens(line)
            if not parts:
                continue
            keyword = parts[0].lower()
            values = parts[1:]
            if "=" in parts[0]:
                raw_keyword, raw_value = parts[0].split("=", 1)
                keyword = raw_keyword.lower()
                values = ([raw_value] if raw_value else []) + parts[1:]
            if keyword == "host":
                aliases.update(value for value in values if _literal_alias(value))
            elif keyword == "include":
                for pattern in values:
                    expanded = os.path.expanduser(pattern.replace("%d", str(Path.home())))
                    candidate = Path(expanded)
                    if not candidate.is_absolute():
                        candidate = include_base / candidate
                    for match in sorted(glob.glob(str(candidate))):
                        visit(Path(match))

    visit(config)
    return sorted(aliases)


def resolve_alias(alias: str, *, config: Path, ssh: str) -> dict[str, str]:
    """Return the small, non-secret subset needed from `ssh -G`."""

    result = subprocess.run(
        [ssh, "-F", str(config), "-G", alias],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"ssh -G failed for {alias}: {detail}")

    effective: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition(" ")
        if separator and key in {"hostname", "port", "user"} and key not in effective:
            effective[key] = value.strip()
    missing = {"hostname", "port", "user"} - effective.keys()
    if missing:
        raise RuntimeError(f"ssh -G omitted {', '.join(sorted(missing))} for {alias}")
    return {
        "alias": alias,
        "hostname": effective["hostname"].lower(),
        "port": effective["port"],
        "user": effective["user"],
    }


def discover(
    *, config: Path, anchor: str, explicit_aliases: list[str], ssh: str
) -> dict[str, object]:
    config = config.expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"SSH config does not exist: {config}")

    anchor_record = resolve_alias(anchor, config=config, ssh=ssh)
    endpoint = (anchor_record["hostname"], anchor_record["port"])

    ordered_candidates = list(dict.fromkeys([anchor, *explicit_aliases, *collect_aliases(config)]))
    by_identity: dict[tuple[str, str, str], dict[str, str]] = {}
    errors: list[dict[str, str]] = []
    for alias in ordered_candidates:
        try:
            record = resolve_alias(alias, config=config, ssh=ssh)
        except RuntimeError as exc:
            errors.append({"alias": alias, "error": str(exc)})
            continue
        if (record["hostname"], record["port"]) != endpoint:
            continue
        identity = (record["hostname"], record["port"], record["user"])
        by_identity.setdefault(identity, record)

    users = sorted(by_identity.values(), key=lambda item: (item["user"], item["alias"]))
    if not users:
        raise RuntimeError(f"no SSH users resolve to the anchor endpoint for {anchor}")
    return {
        "config": str(config),
        "anchor": anchor_record,
        "users": users,
        "resolution_errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", default="bgu-slurm", help="SSH alias defining the cluster endpoint")
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        dest="aliases",
        help="additional candidate alias; repeat as needed",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("~/.ssh/config"),
        help="SSH user config to inspect (default: ~/.ssh/config)",
    )
    parser.add_argument("--ssh", default="ssh", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = discover(
            config=args.config,
            anchor=args.anchor,
            explicit_aliases=args.aliases,
            ssh=args.ssh,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
