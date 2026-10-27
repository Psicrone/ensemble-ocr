"""Matching extracted values against OCR text (strings, numbers, dates)."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

_NUMBER = re.compile(r"[-(]?\d[\d.,']*\d|\d")
_DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%d-%m-%Y", "%Y/%m/%d",
    "%d/%m/%y", "%m/%d/%y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%b %d %Y",
)  # fmt: skip
_DATE_TOKEN = re.compile(
    r"\d{1,4}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{1,2} [A-Za-z]{3,9},? \d{4}|[A-Za-z]{3,9} \d{1,2},? \d{4}"
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w]+", " ", text)
    return " ".join(text.split())


def parse_number(token: str) -> float | None:
    """Parse '1,234.50', '1.234,50', "1'234.50", '(12.00)' and friends."""
    token = token.strip().replace(" ", "").replace("'", "")
    negative = token.startswith(("-", "(")) or token.endswith(")")
    token = token.strip("-()")
    if not token:
        return None
    last_dot, last_comma = token.rfind("."), token.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        decimal = "." if last_dot > last_comma else ","
    elif last_comma >= 0:
        # "1,234" is a thousands separator, "12,50" is a decimal comma.
        decimal = "," if len(token) - last_comma - 1 != 3 else ""
    elif token.count(".") == 1:
        decimal = "."
    else:
        decimal = ""
    thousands = {".", ","} - {decimal}
    for sep in thousands:
        token = token.replace(sep, "")
    if decimal:
        token = token.replace(decimal, ".")
    try:
        value = float(token)
    except ValueError:
        return None
    return -value if negative else value


@lru_cache(maxsize=64)
def numbers_in(text: str) -> tuple[float, ...]:
    values = (parse_number(m.group()) for m in _NUMBER.finditer(text))
    return tuple(v for v in values if v is not None)


def parse_date(token: str) -> date | None:
    token = token.strip().replace(",", ", ").replace(",  ", ", ")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            continue
    return None


@lru_cache(maxsize=64)
def dates_in(text: str) -> tuple[date, ...]:
    values = (parse_date(m.group()) for m in _DATE_TOKEN.finditer(text))
    return tuple(v for v in values if v is not None)


def best_span(needle: str, lines: list[str]) -> tuple[float, str]:
    """Highest similarity of ``needle`` to any line (or window within a line)."""
    target = normalize(needle)
    best = (0.0, "")
    for line in lines:
        norm = normalize(line)
        if not norm:
            continue
        if f" {target} " in f" {norm} ":
            return 1.0, line
        words, size = norm.split(), max(1, len(target.split()))
        windows = {" ".join(words[i : i + size]) for i in range(max(1, len(words) - size + 1))}
        windows.add(norm)
        score = max(SequenceMatcher(None, target, w).ratio() for w in windows)
        if score > best[0]:
            best = (score, line)
    return best


def value_in_text(value: Any, text: str, lines: list[str], threshold: float) -> tuple[bool, float, str | None]:
    """Is ``value`` supported by the OCR text? Returns (found, score, closest_ocr_line)."""
    if isinstance(value, bool) or value is None:
        return True, 1.0, None
    if isinstance(value, (int, float)):
        found = any(abs(n - value) < 0.005 for n in numbers_in(text))
        return found, 1.0 if found else 0.0, None
    if isinstance(value, (date, datetime)):
        value = value.isoformat()[:10]
    value = str(value)
    as_date = parse_date(value) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) else None
    if as_date is not None:
        found = as_date in dates_in(text)
        return found, 1.0 if found else 0.0, None
    needle = normalize(value)
    if needle and f" {needle} " in f" {normalize(text)} ":
        return True, 1.0, None
    score, line = best_span(value, lines)
    return score >= threshold, score, line or None


def matches(value: Any, text: str, threshold: float = 0.85) -> tuple[bool, float]:
    """Does ``text`` (e.g. the cited OCR tokens) say ``value``?

    Numbers and dates must match exactly after parsing; strings are compared
    after normalisation, with a fuzzy ratio for names and addresses.
    """
    if value is None or isinstance(value, bool):
        return True, 1.0
    if isinstance(value, (int, float)):
        ok = any(abs(n - value) < 0.005 for n in numbers_in(text))
        return ok, float(ok)
    if isinstance(value, (date, datetime)):
        value = value.isoformat()[:10]
    value = str(value)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) and (as_date := parse_date(value)):
        ok = as_date in dates_in(text)
        return ok, float(ok)
    needle, hay = normalize(value), normalize(text)
    if not needle:
        return True, 1.0
    if f" {needle} " in f" {hay} ":
        return True, 1.0
    if any(ch.isdigit() for ch in needle):
        return False, 0.0  # IDs, codes and amounts must match exactly; fuzzy is for names
    score = SequenceMatcher(None, needle, hay).ratio()
    return score >= threshold, score
