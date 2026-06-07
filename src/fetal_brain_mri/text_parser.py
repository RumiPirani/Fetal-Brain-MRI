"""Parse fetal brain MRI measurements from free-form text files.

Supports:
  • key = value  /  key: value  (one per line, with or without mm/° units)
  • Two-column CSV  (parameter, value)
  • Header-row CSV  (ga, tcd, atrium_r, … as column names)
  • Free-text lines containing "Label: 33.0 mm" patterns
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Parameter alias table — maps every recognised surface form to a canonical ID
# ---------------------------------------------------------------------------
_ALIASES: dict[str, str] = {}


def _reg(canonical: str, *names: str) -> None:
    for n in names:
        _ALIASES[n.lower().strip()] = canonical


_reg("ga",
     "ga", "gestational age", "gestational_age", "gest age", "gest_age",
     "ga_weeks", "ga weeks")

_reg("skull_bpd",
     "skull bpd", "skull_bpd", "sbpd", "bpd skull",
     "skull biparietal", "skull biparietal diameter")

_reg("skull_ofd",
     "skull ofd", "skull_ofd", "sofd", "ofd skull",
     "skull occipitofrontal", "skull occipito-frontal")

_reg("brain_bpd",
     "brain bpd", "brain_bpd", "bpd", "biparietal diameter",
     "brain biparietal", "max brain width", "brain width")

_reg("brain_ofd_l",
     "brain ofd l", "brain ofd left", "brain_ofd_l",
     "left brain ofd", "ofd left", "ofd l")

_reg("brain_ofd_r",
     "brain ofd r", "brain ofd right", "brain_ofd_r",
     "right brain ofd", "ofd right", "ofd r")

_reg("atrium_r",
     "atrium r", "atrium right", "atrium_r",
     "right atrium", "right lateral ventricle", "right ventricular atrium",
     "rv", "right atrial diameter", "atrial diameter right",
     "right atrial", "right ventricle atrium")

_reg("atrium_l",
     "atrium l", "atrium left", "atrium_l",
     "left atrium", "left lateral ventricle", "left ventricular atrium",
     "lv", "left atrial diameter", "atrial diameter left",
     "left atrial", "left ventricle atrium")

_reg("csp",
     "csp", "csp width", "cavum septum pellucidum",
     "cavum sp", "cavum septi pellucidi", "csp_width")

_reg("cc_length",
     "cc", "cc length", "cc_length", "corpus callosum",
     "corpus callosum length", "cc len")

_reg("tcd",
     "tcd", "transcerebellar diameter", "transcerebellar",
     "cerebellar diameter", "trans cerebellar diameter")

_reg("vermis_cc",
     "vermis cc", "vermis_cc", "vermis height",
     "vermis cranio-caudal", "vermis craniocaudal",
     "cerebellar vermis height", "vermis h")

_reg("vermis_ap",
     "vermis ap", "vermis_ap", "vermis ap diameter",
     "vermis anteroposterior", "vermis antero-posterior")

_reg("tva_degrees",
     "tva", "tva degrees", "tva_degrees",
     "tegmento-vermian angle", "tegmento vermian angle",
     "tegmento vermian", "tegmentovermian angle")

_reg("cisterna_magna_depth",
     "cisterna magna", "cisterna magna depth", "cisterna_magna_depth",
     "cm depth", "cisterna magna d")

_reg("pons_ap",
     "pons", "pons ap", "pons_ap",
     "pons diameter", "pons ap diameter", "pons anteroposterior")

_reg("third_ventricle",
     "third ventricle", "third_ventricle", "3rd ventricle", "3v",
     "third v width", "third ventricle width")

_reg("tdpf",
     "tdpf", "posterior fossa diameter", "transverse posterior fossa",
     "max transverse posterior fossa", "posterior fossa transverse")

_reg("csa",
     "csa", "clivus supraocciput angle", "clivus-supraocciput angle",
     "clivus angle", "clivus supraocciput")

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_VALUE_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_UNIT_STRIP = re.compile(r"\s*(mm|cm|°|deg|degrees)\s*$", re.IGNORECASE)


def _clean_key(raw: str) -> str:
    return re.sub(r"[_\-\s]+", " ", raw).strip().lower()


def _parse_float(raw: str) -> float | None:
    raw = _UNIT_STRIP.sub("", raw).strip().replace(",", ".")
    m = _VALUE_RE.search(raw)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def _resolve(key: str) -> str | None:
    k = _clean_key(key)
    return _ALIASES.get(k)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    values: dict[str, str]          # canonical_id → raw string value (GA stays as string)
    unrecognized: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_text(text: str) -> ParseResult:
    """Attempt to extract measurement values from *text* in any supported format."""

    text = text.strip()
    if not text:
        return ParseResult(values={})

    # Try CSV (header-row or two-column) first if commas/tabs present
    if "," in text or "\t" in text:
        result = _try_csv(text)
        if result is not None:
            return result

    # Run kv-lines and free-text extraction, then merge.
    # kv-lines is more precise (explicit separators) so it wins on conflicts.
    # Free-text fills in whatever kv-lines missed.
    kv = _try_kv_lines(text)
    free = _try_free_text(text)
    merged = {**free.values, **kv.values}   # kv overwrites free on same key
    unrecognized = list(dict.fromkeys(kv.unrecognized + free.unrecognized))
    return ParseResult(values=merged, unrecognized=unrecognized)


# ---------------------------------------------------------------------------
# Format-specific parsers
# ---------------------------------------------------------------------------

def _try_csv(text: str) -> ParseResult | None:
    delim = "\t" if "\t" in text else ","
    try:
        reader = list(csv.reader(io.StringIO(text), delimiter=delim))
    except Exception:
        return None

    if len(reader) < 1:
        return None

    # Case 1: two-column  (parameter, value) — possibly with a header row
    if all(len(row) == 2 for row in reader if row):
        values: dict[str, str] = {}
        unrecognized: list[str] = []
        for row in reader:
            key_raw, val_raw = row[0].strip(), row[1].strip()
            # Skip header row where value looks non-numeric
            if _parse_float(val_raw) is None and key_raw.lower() in ("parameter", "param", "field", "name"):
                continue
            canonical = _resolve(key_raw)
            v = val_raw.strip()
            if canonical:
                values[canonical] = v if canonical == "ga" else _UNIT_STRIP.sub("", v).strip()
            elif key_raw and v:
                unrecognized.append(f"{key_raw}: {v}")
        if values:
            return ParseResult(values=values, unrecognized=unrecognized)

    # Case 2: header row with many columns, data on next row(s)
    if len(reader) >= 2:
        header = [h.strip() for h in reader[0]]
        for data_row in reader[1:]:
            if len(data_row) != len(header):
                continue
            values = {}
            unrecognized = []
            for h, v in zip(header, data_row):
                v = v.strip()
                if not v:
                    continue
                canonical = _resolve(h)
                if canonical:
                    values[canonical] = v
                elif h:
                    unrecognized.append(f"{h}: {v}")
            if values:
                return ParseResult(values=values, unrecognized=unrecognized)

    return None


def _try_kv_lines(text: str) -> ParseResult:
    values: dict[str, str] = {}
    unrecognized: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Split on first = or :
        for sep in ("=", ":"):
            if sep in line:
                key_raw, _, val_raw = line.partition(sep)
                key_raw = key_raw.strip()
                val_raw = val_raw.strip()
                if not key_raw or not val_raw:
                    break
                # val must look numeric (except GA)
                canonical = _resolve(key_raw)
                if canonical == "ga":
                    values["ga"] = val_raw  # keep "28+3" / "28.4" as-is
                    break
                # Take only the first whitespace token so "33.0 mm" works
                # and "33.0  Atrium R: 8.5 …" (multi-kv on one line) doesn't
                # swallow the rest of the line.
                first_tok = val_raw.strip().split()[0] if val_raw.strip() else ""
                cleaned = _UNIT_STRIP.sub("", first_tok).strip()
                if _parse_float(cleaned) is not None:
                    if canonical:
                        values[canonical] = cleaned
                    else:
                        unrecognized.append(f"{key_raw}: {cleaned}")
                break

    return ParseResult(values=values, unrecognized=unrecognized)


_GA_FREE_RE = re.compile(
    r"(?:ga|gestational\s+age)[^\d\n]{0,10}(\d{1,2}[+w]\d{1,2}|\d{2,3}(?:[.,]\d)?)",
    re.IGNORECASE,
)


def _try_free_text(text: str) -> ParseResult:
    """Scan free text for 'Label … numeric_value (unit)' patterns."""
    values: dict[str, str] = {}
    unrecognized: list[str] = []

    # Handle GA specially — supports "28+3" / "28w3" / "28.4" formats
    ga_m = _GA_FREE_RE.search(text)
    if ga_m:
        values["ga"] = ga_m.group(1).replace("w", "+")

    # Pattern: any alias phrase followed by optional punctuation then a number
    all_aliases = sorted(_ALIASES.keys(), key=len, reverse=True)  # longest first
    for alias in all_aliases:
        if alias in ("ga", "gestational age", "gestational_age", "gest age",
                     "gest_age", "ga_weeks", "ga weeks"):
            continue  # already handled above
        # Word-boundary guards prevent "bpd" from matching inside "skull_bpd"
        escaped = re.escape(alias)
        pattern = re.compile(
            r"(?<!\w)" + escaped + r"(?!\w)" +
            r"[^\d\n]{0,20}([-+]?\d+(?:[.,]\d+)?)\s*(?:mm|cm|°|deg)?",
            re.IGNORECASE,
        )
        m = pattern.search(text)
        if m:
            canonical = _ALIASES[alias]
            if canonical not in values:  # first match wins
                values[canonical] = m.group(1).replace(",", ".")

    return ParseResult(values=values, unrecognized=unrecognized)
