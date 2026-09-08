#!/usr/bin/env python3
"""Filter ALLOCATE_ONLY scheduler logs (role=prefill) and pretty-print them.

Typical log line produced by scheduler_server:

    ALLOCATE_ONLY selected req_id=... role=prefill ins=... ep=... score=...
    isl=... prefill_cost=... cpu_hit_tokens=... fast_path=...
    endpoints[ins/ep:running/workload/prefill_cost/cpu_hit_tokens]=...
    avg_active_tokens=... avg_prefill_cost=... avg_cpu_hit_tokens=...

Usage:
    python3 scripts/filter_prefill_allocate_logs.py scheduler.log
    python3 scripts/filter_prefill_allocate_logs.py scheduler.log.gz --cards
    grep ALLOCATE_ONLY app.log | python3 scripts/filter_prefill_allocate_logs.py
    python3 scripts/filter_prefill_allocate_logs.py --demo
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TextIO

ALLOCATE_MARKER = "ALLOCATE_ONLY selected"

# role.value of PDRole.ROLE_P
DEFAULT_ROLE = "prefill"
ROLE_ALIASES = {
    "prefill": "prefill",
    "p": "prefill",
    "role_p": "prefill",
    "decode": "decode",
    "d": "decode",
    "role_d": "decode",
    "encode": "encode",
    "e": "encode",
    "role_e": "encode",
    "union": "union",
    "u": "union",
    "both": "union",
    "role_u": "union",
}

ENDPOINT_TOKEN_RE = re.compile(
    r"(?P<ins>[^\s/:]+)/(?P<ep>[^\s:]+):(?P<metrics>[0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)*)"
)
KV_RE = re.compile(r"(?P<key>[A-Za-z_][\w.\[\];:/]*)=(?P<value>\S+)")
ENDPOINTS_KEY_RE = re.compile(r"endpoints\[[^\]]*\]=")
TRAILING_AVG_RE = re.compile(
    r"\s+(avg_active_tokens|avg_prefill_cost|avg_cpu_hit_tokens)="
)
TIMESTAMP_RES = (
    re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    ),
    re.compile(
        r"(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)"
    ),
)
JSON_MESSAGE_KEYS = ("message", "msg", "log", "text", "event")

BAR_FILL = "█"
BAR_EMPTY = "░"
STAR = "★"


def _east_asian_width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1


def display_width(text: str) -> int:
    return sum(_east_asian_width(ch) for ch in _strip_ansi(text))


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def pad(text: str, width: int, align: str = "left") -> str:
    gap = max(0, width - display_width(text))
    if align == "right":
        return " " * gap + text
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


def truncate(text: str, width: int) -> str:
    if display_width(text) <= width:
        return text
    ellipsis = "…"
    budget = max(1, width - display_width(ellipsis))
    out: list[str] = []
    used = 0
    for ch in text:
        w = _east_asian_width(ch)
        if used + w > budget:
            break
        out.append(ch)
        used += w
    return "".join(out) + ellipsis


def fmt_num(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return f"{int(value)}"
    return f"{value:.{digits}f}"


def parse_number(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.lower() in {"none", "null", "nan", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    text = raw.strip().lower()
    if text in {"true", "yes", "1", "y"}:
        return True
    if text in {"false", "no", "0", "n"}:
        return False
    return None


def normalize_role(raw: str | None) -> str:
    if not raw:
        return ""
    return ROLE_ALIASES.get(raw.strip().lower(), raw.strip().lower())


class Palette:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _c(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def bold(self, text: str) -> str:
        return self._c("1", text)

    def dim(self, text: str) -> str:
        return self._c("2", text)

    def cyan(self, text: str) -> str:
        return self._c("36", text)

    def green(self, text: str) -> str:
        return self._c("32", text)

    def yellow(self, text: str) -> str:
        return self._c("33", text)

    def magenta(self, text: str) -> str:
        return self._c("35", text)

    def red(self, text: str) -> str:
        return self._c("31", text)

    def blue(self, text: str) -> str:
        return self._c("34", text)

    def white(self, text: str) -> str:
        return self._c("97", text)

    def header(self, text: str) -> str:
        return self._c("1;96", text)

    def selected(self, text: str) -> str:
        return self._c("1;32", text)

    def warn(self, text: str) -> str:
        return self._c("1;33", text)


@dataclass
class EndpointSnap:
    instance_id: str
    endpoint_id: str
    running: float | None = None
    active_tokens: float | None = None
    prefill_cost: float | None = None
    cpu_hit_tokens: float | None = None

    @property
    def key(self) -> str:
        return f"{self.instance_id}/{self.endpoint_id}"

    def as_dict(self) -> dict[str, object]:
        return {
            "ins": self.instance_id,
            "ep": self.endpoint_id,
            "running": self.running,
            "active_tokens": self.active_tokens,
            "prefill_cost": self.prefill_cost,
            "cpu_hit_tokens": self.cpu_hit_tokens,
        }


@dataclass
class AllocateRecord:
    raw: str
    timestamp: str | None = None
    req_id: str = ""
    role: str = ""
    instance_id: str = ""
    endpoint_id: str = ""
    score: float | None = None
    isl: float | None = None
    prefill_cost: float | None = None
    cpu_hit_tokens: float | None = None
    fast_path: bool | None = None
    endpoints: list[EndpointSnap] = field(default_factory=list)
    avg_active_tokens: float | None = None
    avg_prefill_cost: float | None = None
    avg_cpu_hit_tokens: float | None = None
    source: str = ""
    line_no: int = 0

    @property
    def selected_key(self) -> str:
        return f"{self.instance_id}/{self.endpoint_id}"

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "req_id": self.req_id,
            "role": self.role,
            "ins": self.instance_id,
            "ep": self.endpoint_id,
            "score": self.score,
            "isl": self.isl,
            "prefill_cost": self.prefill_cost,
            "cpu_hit_tokens": self.cpu_hit_tokens,
            "fast_path": self.fast_path,
            "avg_active_tokens": self.avg_active_tokens,
            "avg_prefill_cost": self.avg_prefill_cost,
            "avg_cpu_hit_tokens": self.avg_cpu_hit_tokens,
            "endpoints": [ep.as_dict() for ep in self.endpoints],
            "source": self.source,
            "line_no": self.line_no,
        }


def _maybe_unwrap_json(line: str) -> str:
    stripped = line.strip()
    if not stripped.startswith("{"):
        return line
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return line
    if not isinstance(payload, dict):
        return line
    for key in JSON_MESSAGE_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and ALLOCATE_MARKER in value:
            return value
    # Some loggers put the whole event in a nested "record" / "extra".
    for nested_key in ("record", "extra", "data"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            for key in JSON_MESSAGE_KEYS:
                value = nested.get(key)
                if isinstance(value, str) and ALLOCATE_MARKER in value:
                    return value
    return line


def _extract_timestamp(line: str) -> str | None:
    for pattern in TIMESTAMP_RES:
        match = pattern.search(line)
        if match:
            return match.group("ts").replace(",", ".")
    return None


def _split_endpoints_blob(message: str) -> tuple[str, str]:
    """Return (message_without_endpoints_values, endpoints_blob)."""
    match = ENDPOINTS_KEY_RE.search(message)
    if not match:
        return message, ""
    start = match.end()
    rest = message[start:]
    trail = TRAILING_AVG_RE.search(rest)
    if trail:
        blob = rest[: trail.start()].strip()
        kept = message[:start] + message[start + trail.start() :]
        return kept, blob
    blob = rest.strip()
    return message[:start], blob


def _parse_endpoint_tokens(blob: str) -> list[EndpointSnap]:
    if not blob or blob == "<none>":
        return []
    endpoints: list[EndpointSnap] = []
    for match in ENDPOINT_TOKEN_RE.finditer(blob):
        metrics = [parse_number(part) for part in match.group("metrics").split("/")]
        snap = EndpointSnap(
            instance_id=match.group("ins"),
            endpoint_id=match.group("ep"),
            running=metrics[0] if len(metrics) > 0 else None,
            active_tokens=metrics[1] if len(metrics) > 1 else None,
            prefill_cost=metrics[2] if len(metrics) > 2 else None,
            cpu_hit_tokens=metrics[3] if len(metrics) > 3 else None,
        )
        endpoints.append(snap)
    return endpoints


def parse_allocate_line(line: str, source: str = "", line_no: int = 0) -> AllocateRecord | None:
    text = _maybe_unwrap_json(line.rstrip("\n"))
    marker_at = text.find(ALLOCATE_MARKER)
    if marker_at < 0:
        return None
    timestamp = _extract_timestamp(text[:marker_at] + " " + text)
    message = text[marker_at + len(ALLOCATE_MARKER) :].strip()
    message, endpoints_blob = _split_endpoints_blob(message)

    fields: dict[str, str] = {}
    for match in KV_RE.finditer(message):
        key = match.group("key")
        if key.startswith("endpoints["):
            continue
        fields[key] = match.group("value")

    record = AllocateRecord(
        raw=line.rstrip("\n"),
        timestamp=timestamp,
        req_id=fields.get("req_id", ""),
        role=normalize_role(fields.get("role")),
        instance_id=fields.get("ins", ""),
        endpoint_id=fields.get("ep", ""),
        score=parse_number(fields.get("score")),
        isl=parse_number(fields.get("isl")),
        prefill_cost=parse_number(fields.get("prefill_cost")),
        cpu_hit_tokens=parse_number(fields.get("cpu_hit_tokens")),
        fast_path=parse_bool(fields.get("fast_path")),
        endpoints=_parse_endpoint_tokens(endpoints_blob),
        avg_active_tokens=parse_number(fields.get("avg_active_tokens")),
        avg_prefill_cost=parse_number(fields.get("avg_prefill_cost")),
        avg_cpu_hit_tokens=parse_number(fields.get("avg_cpu_hit_tokens")),
        source=source,
        line_no=line_no,
    )
    return record


def _open_log(path: Path) -> TextIO:
    if path.suffix == ".gz" or path.name.endswith(".log.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_records(paths: Sequence[Path], stdin: TextIO) -> Iterator[AllocateRecord]:
    if not paths:
        for line_no, line in enumerate(stdin, 1):
            record = parse_allocate_line(line, source="<stdin>", line_no=line_no)
            if record is not None:
                yield record
        return
    for path in paths:
        with _open_log(path) as handle:
            for line_no, line in enumerate(handle, 1):
                record = parse_allocate_line(line, source=str(path), line_no=line_no)
                if record is not None:
                    yield record


def filter_records(
    records: Iterable[AllocateRecord],
    role: str,
    limit: int | None,
) -> list[AllocateRecord]:
    wanted = normalize_role(role)
    out: list[AllocateRecord] = []
    for record in records:
        if wanted and record.role != wanted:
            continue
        out.append(record)
        if limit is not None and len(out) >= limit:
            break
    return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _bar(ratio: float, width: int) -> str:
    ratio = 0.0 if math.isnan(ratio) else min(max(ratio, 0.0), 1.0)
    filled = int(round(ratio * width))
    filled = min(max(filled, 0), width)
    return BAR_FILL * filled + BAR_EMPTY * (width - filled)


def _term_width() -> int:
    return max(80, shutil.get_terminal_size((120, 24)).columns)


def _rule(width: int, char: str = "─") -> str:
    return char * width


def render_banner(color: Palette, records: Sequence[AllocateRecord], role: str, width: int) -> list[str]:
    times = [r.timestamp for r in records if r.timestamp]
    span = f"{times[0]}  →  {times[-1]}" if times else "时间未知"
    title = f" ALLOCATE_ONLY  ·  role={role}  ·  {len(records)} 条 "
    line = [
        color.header("╔" + "═" * (width - 2) + "╗"),
        color.header("║") + color.bold(pad(title, width - 2, "center")) + color.header("║"),
        color.header("║") + color.dim(pad(f"  {span}", width - 2)) + color.header("║"),
        color.header("╚" + "═" * (width - 2) + "╝"),
    ]
    return line


def render_summary(color: Palette, records: Sequence[AllocateRecord], width: int) -> list[str]:
    if not records:
        return [color.yellow("没有匹配 role=prefill 的 ALLOCATE_ONLY 记录。")]

    scores = [r.score for r in records if r.score is not None]
    isls = [r.isl for r in records if r.isl is not None]
    prefills = [r.prefill_cost for r in records if r.prefill_cost is not None]
    hits = [r.cpu_hit_tokens for r in records if r.cpu_hit_tokens is not None]
    fast = [r.fast_path for r in records if r.fast_path is not None]
    selected = Counter(r.selected_key for r in records if r.instance_id)

    lines = ["", color.bold("▸ 调度概览"), color.dim(_rule(min(width, 88)))]

    def kv(label: str, value: str) -> str:
        return f"  {color.cyan(pad(label, 16))} {color.white(value)}"

    lines.append(kv("记录数", str(len(records))))
    lines.append(kv("唯一 req", str(len({r.req_id for r in records if r.req_id}))))
    lines.append(kv("平均 score", fmt_num(_mean(scores), 4) if scores else "—"))
    lines.append(kv("平均 ISL", fmt_num(_mean(isls)) if isls else "—"))
    lines.append(kv("平均 prefill", fmt_num(_mean(prefills)) if prefills else "—"))
    lines.append(kv("平均 cpu_hit", fmt_num(_mean(hits)) if hits else "—"))
    if fast:
        rate = sum(1 for v in fast if v) / len(fast)
        lines.append(kv("fast_path", f"{sum(1 for v in fast if v)}/{len(fast)}  ({rate:.1%})"))

    if selected:
        lines += ["", color.bold("▸ 选中分布  ins/ep"), color.dim(_rule(min(width, 88)))]
        max_count = max(selected.values())
        bar_w = min(28, max(12, width - 42))
        for key, count in selected.most_common():
            ratio = count / len(records)
            bar = _bar(count / max_count, bar_w)
            label = pad(key, 12)
            lines.append(
                f"  {color.green(label)} {color.cyan(bar)}  "
                f"{color.bold(str(count))} {color.dim(f'({ratio:.1%})')}"
            )
    return lines


def _table_columns(records: Sequence[AllocateRecord]) -> list[tuple[str, str, int, str]]:
    """Return (key, header, width, align)."""
    req_w = min(36, max(12, max((len(r.req_id) for r in records), default=12)))
    ins_w = max(7, max((len(r.selected_key) for r in records), default=7))
    return [
        ("idx", "#", 4, "right"),
        ("ts", "时间", 14, "left"),
        ("req", "req_id", req_w, "left"),
        ("sel", "选中", ins_w, "left"),
        ("score", "score", 8, "right"),
        ("isl", "ISL", 7, "right"),
        ("prefill", "prefill", 9, "right"),
        ("hit", "cpu_hit", 9, "right"),
        ("fast", "fast", 5, "center"),
        ("avg_a", "avg_tok", 8, "right"),
        ("avg_p", "avg_pc", 8, "right"),
        ("avg_h", "avg_hit", 8, "right"),
    ]


def _short_time(ts: str | None) -> str:
    if not ts:
        return "—"
    # Keep HH:MM:SS.mmm when possible.
    match = re.search(r"(\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?)", ts)
    return match.group(1) if match else ts


def render_table(
    color: Palette,
    records: Sequence[AllocateRecord],
    width: int,
    show_snapshot: bool,
) -> list[str]:
    if not records:
        return []
    cols = _table_columns(records)

    def _table_width(items: list[tuple[str, str, int, str]]) -> int:
        return sum(w for _, _, w, _ in items) + 2 * max(0, len(items) - 1)

    if _table_width(cols) > width:
        cols = [c for c in cols if c[0] not in {"avg_a", "avg_p", "avg_h"}]
    overflow = _table_width(cols) - width
    if overflow > 0:
        cols = [
            (k, h, max(10, w - overflow) if k == "req" else w, a)
            for k, h, w, a in cols
        ]

    def cells(row: dict[str, str], header: bool = False) -> str:
        parts: list[str] = []
        for key, _, w, align in cols:
            text = truncate(row.get(key, ""), w)
            text = pad(text, w, align)
            if header:
                text = color.cyan(color.bold(text))
            parts.append(text)
        return "  ".join(parts)

    header_row = {key: title for key, title, _, _ in cols}
    lines = [
        "",
        color.bold("▸ 逐条调度"),
        color.dim(_rule(min(width, 88))),
        cells(header_row, header=True),
        color.dim("  ".join("─" * w for _, _, w, _ in cols)),
    ]

    for idx, rec in enumerate(records, 1):
        row = {
            "idx": str(idx),
            "ts": _short_time(rec.timestamp),
            "req": rec.req_id or "—",
            "sel": rec.selected_key if rec.instance_id else "—",
            "score": fmt_num(rec.score, 4),
            "isl": fmt_num(rec.isl, 0),
            "prefill": fmt_num(rec.prefill_cost),
            "hit": fmt_num(rec.cpu_hit_tokens),
            "fast": "✓" if rec.fast_path else ("·" if rec.fast_path is False else "—"),
            "avg_a": fmt_num(rec.avg_active_tokens),
            "avg_p": fmt_num(rec.avg_prefill_cost),
            "avg_h": fmt_num(rec.avg_cpu_hit_tokens),
        }
        painted = cells(row)
        if rec.fast_path:
            painted = color.yellow(painted)
        lines.append(painted)
        if show_snapshot:
            lines.extend(_render_snapshot_oneline(color, rec, width))
    return lines


def _render_snapshot_oneline(color: Palette, rec: AllocateRecord, width: int) -> list[str]:
    if not rec.endpoints:
        return [color.dim("     endpoints  <none>")]
    parts: list[str] = []
    for ep in rec.endpoints:
        token = (
            f"{ep.key}:{fmt_num(ep.running, 0)}/"
            f"{fmt_num(ep.active_tokens)}/"
            f"{fmt_num(ep.prefill_cost)}/"
            f"{fmt_num(ep.cpu_hit_tokens)}"
        )
        if ep.key == rec.selected_key:
            parts.append(color.selected(f"{STAR}{token}"))
        else:
            parts.append(color.dim(token))
    prefix = "     "
    body = "  ".join(parts)
    avg = color.dim(
        f"  avg(tok={fmt_num(rec.avg_active_tokens)}  "
        f"pc={fmt_num(rec.avg_prefill_cost)}  "
        f"hit={fmt_num(rec.avg_cpu_hit_tokens)})"
    )
    line = prefix + body + avg
    if display_width(_strip_ansi(line)) <= width:
        return [line]
    # Wrap endpoint tokens if the terminal is narrow.
    wrapped = [prefix + color.dim("endpoints")]
    chunk: list[str] = []
    used = 0
    indent = "       "
    budget = max(20, width - display_width(indent))
    for part in parts:
        w = display_width(_strip_ansi(part)) + 2
        if chunk and used + w > budget:
            wrapped.append(indent + "  ".join(chunk))
            chunk = [part]
            used = w
        else:
            chunk.append(part)
            used += w
    if chunk:
        wrapped.append(indent + "  ".join(chunk))
    wrapped.append(indent + avg.strip())
    return wrapped


def render_cards(color: Palette, records: Sequence[AllocateRecord], width: int) -> list[str]:
    lines = ["", color.bold("▸ 调度卡片"), color.dim(_rule(min(width, 88)))]
    inner = max(60, width - 2)
    for idx, rec in enumerate(records, 1):
        title = (
            f" #{idx}  {_short_time(rec.timestamp)}  "
            f"req={rec.req_id or '—'}  selected={rec.selected_key or '—'}"
        )
        lines.append(color.cyan("┌" + pad(truncate(title, inner - 2), inner - 2, "left") + "┐"))

        meta = (
            f"  score={fmt_num(rec.score, 4)}   isl={fmt_num(rec.isl, 0)}   "
            f"prefill={fmt_num(rec.prefill_cost)}   cpu_hit={fmt_num(rec.cpu_hit_tokens)}   "
            f"fast_path={fmt_num(rec.fast_path)}"
        )
        lines.append("│" + pad(truncate(meta, inner - 2), inner - 2) + "│")
        avg = (
            f"  cluster avg   active={fmt_num(rec.avg_active_tokens)}   "
            f"prefill={fmt_num(rec.avg_prefill_cost)}   "
            f"cpu_hit={fmt_num(rec.avg_cpu_hit_tokens)}"
        )
        lines.append("│" + pad(color.dim(truncate(avg, inner - 2)), inner - 2) + "│")
        lines.append("│" + " " * (inner - 2) + "│")

        header = (
            f"  {pad('ins/ep', 10)} {pad('run', 4, 'right')}  "
            f"{pad('active', 18)} {pad('prefill', 9, 'right')} {pad('cpu_hit', 9, 'right')}"
        )
        lines.append("│" + pad(color.dim(truncate(header, inner - 2)), inner - 2) + "│")

        if not rec.endpoints:
            lines.append("│" + pad(color.dim("  (no endpoint snapshot)"), inner - 2) + "│")
        else:
            max_active = max((ep.active_tokens or 0.0) for ep in rec.endpoints) or 1.0
            bar_w = min(12, max(6, inner - 52))
            for ep in rec.endpoints:
                mark = STAR if ep.key == rec.selected_key else " "
                bar = _bar((ep.active_tokens or 0.0) / max_active, bar_w)
                row = (
                    f"  {mark}{pad(ep.key, 10)} {pad(fmt_num(ep.running, 0), 4, 'right')}  "
                    f"{bar} {pad(fmt_num(ep.active_tokens), 6, 'right')} "
                    f"{pad(fmt_num(ep.prefill_cost), 9, 'right')} "
                    f"{pad(fmt_num(ep.cpu_hit_tokens), 9, 'right')}"
                )
                if ep.key == rec.selected_key:
                    row = color.selected(truncate(row, inner - 2))
                else:
                    row = truncate(row, inner - 2)
                lines.append("│" + pad(row, inner - 2) + "│")
        lines.append(color.cyan("└" + "─" * (inner - 2) + "┘"))
        lines.append("")
    return lines


def dump_json(records: Sequence[AllocateRecord], handle: TextIO) -> None:
    json.dump([r.as_dict() for r in records], handle, ensure_ascii=False, indent=2)
    handle.write("\n")


def dump_csv(records: Sequence[AllocateRecord], handle: TextIO) -> None:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "timestamp",
            "req_id",
            "role",
            "ins",
            "ep",
            "score",
            "isl",
            "prefill_cost",
            "cpu_hit_tokens",
            "fast_path",
            "avg_active_tokens",
            "avg_prefill_cost",
            "avg_cpu_hit_tokens",
            "endpoints",
        ]
    )
    for rec in records:
        endpoints = " ".join(
            f"{ep.key}:{fmt_num(ep.running, 0)}/{fmt_num(ep.active_tokens)}/"
            f"{fmt_num(ep.prefill_cost)}/{fmt_num(ep.cpu_hit_tokens)}"
            for ep in rec.endpoints
        )
        writer.writerow(
            [
                rec.timestamp or "",
                rec.req_id,
                rec.role,
                rec.instance_id,
                rec.endpoint_id,
                fmt_num(rec.score, 4),
                fmt_num(rec.isl),
                fmt_num(rec.prefill_cost),
                fmt_num(rec.cpu_hit_tokens),
                "" if rec.fast_path is None else str(rec.fast_path).lower(),
                fmt_num(rec.avg_active_tokens),
                fmt_num(rec.avg_prefill_cost),
                fmt_num(rec.avg_cpu_hit_tokens),
                endpoints,
            ]
        )


DEMO_LINES = [
    "2026-09-08 10:00:00,120 INFO scheduler: ALLOCATE_ONLY selected req_id=req-001 role=prefill ins=1 ep=0 score=0.8123 isl=2048 prefill_cost=128.0 cpu_hit_tokens=1024 fast_path=False endpoints[ins/ep:running/workload/prefill_cost/cpu_hit_tokens]=1/0:3/256.0/128.0/2048.0 1/1:1/128.0/64.0/1024.0 2/0:2/192.0/96.0/1536.0 2/1:0/64.0/32.0/512.0 avg_active_tokens=160.0 avg_prefill_cost=80.0 avg_cpu_hit_tokens=1280.0",
    "2026-09-08 10:00:00,340 INFO scheduler: ALLOCATE_ONLY selected req_id=req-002 role=decode ins=3 ep=1 score=0.1000 isl=32 prefill_cost=0.0 cpu_hit_tokens=0 fast_path=True endpoints[ins/ep:running/workload]=3/0:4/40.0 3/1:1/12.0",
    "2026-09-08 10:00:01,018 INFO scheduler: ALLOCATE_ONLY selected req_id=req-003 role=prefill ins=1 ep=1 score=0.7440 isl=1024 prefill_cost=96.5 cpu_hit_tokens=768 fast_path=False endpoints[ins/ep:running/workload/prefill_cost/cpu_hit_tokens]=1/0:4/280.0/140.0/2176.0 1/1:1/96.0/48.0/768.0 2/0:2/192.0/96.0/1536.0 2/1:1/80.0/40.0/640.0 avg_active_tokens=162.0 avg_prefill_cost=81.0 avg_cpu_hit_tokens=1280.0",
    "2026-09-08 10:00:01,501 INFO scheduler: ALLOCATE_ONLY selected req_id=req-004 role=prefill ins=2 ep=0 score=0.9012 isl=4096 prefill_cost=256.0 cpu_hit_tokens=2048 fast_path=True endpoints[ins/ep:running/workload/prefill_cost/cpu_hit_tokens]=1/0:4/280.0/140.0/2176.0 1/1:2/160.0/80.0/1280.0 2/0:2/120.0/60.0/960.0 2/1:1/80.0/40.0/640.0 avg_active_tokens=160.0 avg_prefill_cost=80.0 avg_cpu_hit_tokens=1264.0",
    "2026-09-08 10:00:02,088 INFO scheduler: ALLOCATE_ONLY selected req_id=req-005 role=prefill ins=2 ep=1 score=0.6601 isl=512 prefill_cost=48.0 cpu_hit_tokens=256 fast_path=False endpoints[ins/ep:running/workload/prefill_cost/cpu_hit_tokens]=1/0:5/320.0/160.0/2304.0 1/1:2/160.0/80.0/1280.0 2/0:3/180.0/90.0/1440.0 2/1:1/48.0/24.0/384.0 avg_active_tokens=177.0 avg_prefill_cost=88.5 avg_cpu_hit_tokens=1352.0",
    "2026-09-08 10:00:02,410 INFO scheduler: ALLOCATE_ONLY selected req_id=req-006 role=prefill ins=1 ep=0 score=0.5588 isl=256 prefill_cost=32.0 cpu_hit_tokens=128 fast_path=False endpoints[ins/ep:running/workload/prefill_cost/cpu_hit_tokens]=1/0:5/300.0/150.0/2176.0 1/1:2/160.0/80.0/1280.0 2/0:3/180.0/90.0/1440.0 2/1:2/80.0/40.0/512.0 avg_active_tokens=180.0 avg_prefill_cost=90.0 avg_cpu_hit_tokens=1352.0",
]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从调度日志中筛出 ALLOCATE_ONLY / role=prefill 记录并美化展示。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  %(prog)s coordinator.log\n"
            "  %(prog)s coordinator.log.gz --cards\n"
            "  %(prog)s --compact app.log other.log\n"
            "  grep ALLOCATE_ONLY app.log | %(prog)s\n"
            "  %(prog)s --demo --cards\n"
        ),
    )
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        help="日志文件（支持 .gz）。省略则读 stdin。",
    )
    parser.add_argument(
        "-r",
        "--role",
        default=DEFAULT_ROLE,
        help=f"过滤的 role（默认 {DEFAULT_ROLE}）。传 all 不过滤。",
    )
    parser.add_argument("-n", "--limit", type=int, default=None, help="最多展示前 N 条。")
    parser.add_argument(
        "--cards",
        action="store_true",
        help="每条调度一张卡片，含 endpoint 柱状快照。",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="只打总表，不附带每行 endpoint 快照。",
    )
    parser.add_argument("--summary-only", action="store_true", help="只输出概览统计。")
    parser.add_argument("--json", action="store_true", help="输出 JSON。")
    parser.add_argument("--csv", action="store_true", dest="csv_out", help="输出 CSV。")
    parser.add_argument("--no-color", action="store_true", help="关闭 ANSI 颜色。")
    parser.add_argument("--demo", action="store_true", help="用内置样例日志预览样式。")
    return parser.parse_args(argv)


def _color_enabled(args: argparse.Namespace) -> bool:
    if args.no_color or args.json or args.csv_out:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def collect_records(args: argparse.Namespace) -> list[AllocateRecord]:
    if args.demo:
        parsed = [
            rec
            for line_no, line in enumerate(DEMO_LINES, 1)
            if (rec := parse_allocate_line(line, source="<demo>", line_no=line_no))
        ]
    else:
        missing = [str(path) for path in args.logs if not path.exists()]
        if missing:
            raise FileNotFoundError("找不到日志文件: " + ", ".join(missing))
        parsed = list(iter_records(args.logs, sys.stdin))
    role = "" if args.role.lower() in {"all", "*"} else args.role
    return filter_records(parsed, role=role, limit=args.limit)


def render_pretty(args: argparse.Namespace, records: list[AllocateRecord]) -> str:
    color = Palette(_color_enabled(args))
    width = _term_width()
    role = args.role if args.role.lower() not in {"all", "*"} else "all"
    chunks: list[str] = []
    chunks.extend(render_banner(color, records, role, width))
    chunks.extend(render_summary(color, records, width))
    if args.summary_only:
        return "\n".join(chunks) + "\n"
    if args.cards:
        chunks.extend(render_cards(color, records, width))
    else:
        chunks.extend(render_table(color, records, width, show_snapshot=not args.compact))
        if records and not args.compact:
            chunks.append("")
            chunks.append(color.dim("提示: 绿色 ★ 为本次选中的 ins/ep；加 --cards 可看柱状快照。"))
    return "\n".join(chunks) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        records = collect_records(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        dump_json(records, sys.stdout)
        return 0
    if args.csv_out:
        dump_csv(records, sys.stdout)
        return 0
    sys.stdout.write(render_pretty(args, records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
