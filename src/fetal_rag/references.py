"""Reference manifest parsing from the FetalBrainMRI SPEC."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


URL_RE = re.compile(r"https?://[^\s|]+")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s|;]+", re.IGNORECASE)
PMID_RE = re.compile(r"\bPMID[: ]+(\d+)\b", re.IGNORECASE)
PMCID_RE = re.compile(r"\bPMC\d+\b", re.IGNORECASE)


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in URL_RE.findall(text):
        url = match.rstrip(".,;")
        while url.endswith(")") and url.count("(") < url.count(")"):
            url = url[:-1]
        urls.append(url)
    return urls


def _extract_doi(*parts: str) -> str:
    for part in parts:
        match = DOI_RE.search(part)
        if match:
            return match.group(0).rstrip(".,")
    return ""


def _extract_pmid(*parts: str) -> str:
    for part in parts:
        match = PMID_RE.search(part)
        if match:
            return match.group(1)
    return ""


def _extract_pmcid(*parts: str) -> str:
    for part in parts:
        match = PMCID_RE.search(part)
        if match:
            return match.group(0)
    return ""


def parse_primary_source_inventory(spec_text: str) -> list[dict[str, Any]]:
    """Parse SPEC section 7.2 source inventory rows."""

    marker = "### 7.2 Primary Source Inventory"
    start = spec_text.find(marker)
    if start < 0:
        return []

    next_section = spec_text.find("\n### ", start + len(marker))
    section = spec_text[start:] if next_section < 0 else spec_text[start:next_section]
    entries: list[dict[str, Any]] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = _split_markdown_row(line)
        if len(cells) != 7 or cells[0] in {"Source ID", "---"}:
            continue
        source_id, citation, modality, doi, pmid, pmcid, url_text = cells
        entries.append(
            {
                "source_id": source_id,
                "citation": citation,
                "modality": modality,
                "doi": _extract_doi(doi, citation),
                "pmid": pmid if pmid.isdigit() else _extract_pmid(citation),
                "pmcid": pmcid if pmcid.startswith("PMC") else _extract_pmcid(citation),
                "urls": _extract_urls(url_text),
                "spec_section": "7.2 Primary Source Inventory",
            }
        )
    return entries


def parse_numbered_references(spec_text: str) -> list[dict[str, Any]]:
    """Parse the final numbered reference list from SPEC.md."""

    marker = "## References"
    start = spec_text.find(marker)
    if start < 0:
        return []

    section = spec_text[start:]
    entries: list[dict[str, Any]] = []
    for line in section.splitlines():
        match = re.match(r"^\[(\d+)\]\s+(.+)$", line.strip())
        if not match:
            continue
        number = int(match.group(1))
        citation = match.group(2).strip()
        entries.append(
            {
                "source_id": f"REF_{number:03d}",
                "reference_number": number,
                "citation": citation,
                "doi": _extract_doi(citation),
                "pmid": _extract_pmid(citation),
                "pmcid": _extract_pmcid(citation),
                "urls": _extract_urls(citation),
                "spec_section": "References",
            }
        )
    return entries


def build_manifest_from_spec(spec_path: str | Path) -> dict[str, Any]:
    path = Path(spec_path)
    spec_text = path.read_text(encoding="utf-8")
    primary_sources = parse_primary_source_inventory(spec_text)
    numbered_references = parse_numbered_references(spec_text)
    return {
        "generated_from": str(path),
        "primary_source_count": len(primary_sources),
        "numbered_reference_count": len(numbered_references),
        "entries": primary_sources + numbered_references,
    }


def write_manifest_from_spec(spec_path: str | Path, output_path: str | Path) -> None:
    manifest = build_manifest_from_spec(spec_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return list(data.get("entries", []))


def match_reference_for_pdf(
    pdf_path: str | Path, references: list[dict[str, Any]]
) -> dict[str, Any]:
    """Match a PDF filename to a manifest entry, falling back to the stem."""

    path = Path(pdf_path)
    normalized_stem = re.sub(r"[^a-z0-9]+", "", path.stem.lower())
    for reference in references:
        source_id = str(reference.get("source_id", ""))
        normalized_id = re.sub(r"[^a-z0-9]+", "", source_id.lower())
        if normalized_id and normalized_id in normalized_stem:
            return reference

    return {
        "source_id": path.stem.upper().replace(" ", "_"),
        "citation": path.stem,
        "urls": [],
    }
