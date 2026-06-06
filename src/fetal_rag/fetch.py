"""Open-access PDF fetch helpers for PMC-hosted papers."""

from __future__ import annotations

import json
import http.client
import re
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from html import unescape
from io import BytesIO
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse
from xml.etree import ElementTree

from .references import load_manifest


USER_AGENT = "fetal-brain-mri-rag/0.1 (+https://huggingface.co/spaces)"
FETCH_ERRORS = (
    ElementTree.ParseError,
    http.client.HTTPException,
    OSError,
    TimeoutError,
    ValueError,
    tarfile.TarError,
    urllib.error.URLError,
)


def _ncbi_ftp_https_url(url: str) -> str:
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        return "https://ftp.ncbi.nlm.nih.gov/" + url.removeprefix(
            "ftp://ftp.ncbi.nlm.nih.gov/"
        )
    return url


def _deprecated_ncbi_ftp_url(url: str) -> str:
    prefix = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"
    if not url.startswith(prefix):
        return ""
    remainder = url.removeprefix(prefix)
    if not remainder.startswith(("oa_pdf/", "oa_package/")):
        return ""
    return prefix + "deprecated/" + remainder


@dataclass(frozen=True)
class FetchResult:
    source_id: str
    pmcid: str
    destination: str
    status: str
    message: str = ""


def pmc_pdf_url(pmcid: str) -> str:
    clean = pmcid.strip().upper()
    if not clean.startswith("PMC"):
        raise ValueError("PMCID must start with PMC")
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{clean}/pdf/"


def pmc_article_url(pmcid: str) -> str:
    clean = pmcid.strip().upper()
    if not clean.startswith("PMC"):
        raise ValueError("PMCID must start with PMC")
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{clean}/"


def pmc_oa_url(pmcid: str) -> str:
    clean = pmcid.strip().upper()
    if not clean.startswith("PMC"):
        raise ValueError("PMCID must start with PMC")
    return f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={clean}"


def _extract_pmcid_from_url(url: str) -> str:
    match = re.search(r"/(?:pmc/)?articles/(PMC)?(\d+)", url, re.IGNORECASE)
    if not match:
        return ""
    return "PMC" + match.group(2)


def _download(url: str, timeout: int) -> bytes:
    url = _ncbi_ftp_https_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        deprecated_url = _deprecated_ncbi_ftp_url(url)
        if error.code != 404 or not deprecated_url:
            raise
        fallback_request = urllib.request.Request(
            deprecated_url, headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(fallback_request, timeout=timeout) as response:
            return response.read()


def _validate_pdf(content: bytes) -> bytes:
    if not content.startswith(b"%PDF"):
        raise ValueError("downloaded content was not a PDF")
    return content


def _extract_pdf_from_tgz(content: bytes) -> bytes:
    with tarfile.open(fileobj=BytesIO(content), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".pdf"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            return _validate_pdf(extracted.read())
    raise ValueError("OA package did not contain a PDF")


def _download_pdf_from_article_page(pmcid: str, timeout: int) -> bytes:
    article_url = pmc_article_url(pmcid)
    page = _download(article_url, timeout).decode("utf-8", "replace")
    for match in re.finditer(r"""href=["']([^"']+\.pdf[^"']*)["']""", page):
        pdf_url = urljoin(article_url, unescape(match.group(1)))
        return _validate_pdf(_download(pdf_url, timeout))
    raise ValueError("PMC article page did not expose a PDF link")


def _download_pdf_from_oa_package(pmcid: str, timeout: int) -> bytes:
    xml_content = _download(pmc_oa_url(pmcid), timeout)
    root = ElementTree.fromstring(xml_content)
    links = root.findall(".//link")
    for link in links:
        if link.attrib.get("format") == "pdf" and link.attrib.get("href"):
            return _validate_pdf(_download(link.attrib["href"], timeout))
    for link in links:
        if link.attrib.get("format") == "tgz" and link.attrib.get("href"):
            return _extract_pdf_from_tgz(_download(link.attrib["href"], timeout))
    raise ValueError("PMC OA service did not return a PDF or tgz package link")


def _download_pmc_pdf(pmcid: str, timeout: int) -> bytes:
    try:
        return _download_pdf_from_article_page(pmcid, timeout)
    except FETCH_ERRORS:
        pass
    try:
        return _validate_pdf(_download(pmc_pdf_url(pmcid), timeout))
    except FETCH_ERRORS:
        return _download_pdf_from_oa_package(pmcid, timeout)


def _candidate_pdf_urls(url: str, doi: str = "") -> list[str]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    candidates: list[str] = _doi_pdf_candidates(doi)
    if parsed.path.lower().endswith(".pdf"):
        candidates.append(url)

    if (
        host in {"pmc.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov"}
        and "/articles/" in parsed.path
    ):
        pmcid = _extract_pmcid_from_url(url)
        if pmcid:
            candidates.extend(
                [
                    "https://europepmc.org/backend/ptpmcrender.fcgi?"
                    f"accid={pmcid.upper()}&blobtype=pdf",
                    pmc_pdf_url(pmcid),
                ]
            )

    if host.endswith("nature.com") and "/articles/" in parsed.path:
        candidates.append(url.rstrip("/") + ".pdf")

    if host.endswith("springer.com") and doi:
        candidates.append(
            f"https://link.springer.com/content/pdf/{quote(doi, safe='/')}.pdf"
        )

    if host.endswith("wiley.com") and doi:
        encoded_doi = quote(doi, safe="/")
        candidates.extend(
            [
                f"https://onlinelibrary.wiley.com/doi/pdfdirect/{encoded_doi}",
                f"https://obgyn.onlinelibrary.wiley.com/doi/pdfdirect/{encoded_doi}",
                f"https://onlinelibrary.wiley.com/doi/pdf/{encoded_doi}",
                f"https://obgyn.onlinelibrary.wiley.com/doi/pdf/{encoded_doi}",
            ]
        )

    if host.endswith("ajnr.org") and "/content/" in parsed.path:
        base = url.rstrip("/")
        candidates.extend([base + ".full.pdf", base + ".full.pdf?download=1"])

    if host.endswith("biorxiv.org") and "/content/" in parsed.path:
        candidates.append(url.rstrip("/") + ".full.pdf")

    if host.endswith("mdpi.com"):
        candidates.extend(
            [url.rstrip("/") + "/pdf", url.rstrip("/") + "/pdf?download=1"]
        )

    if host.endswith("frontiersin.org") and "/articles/" in parsed.path:
        candidates.append(url.rstrip("/") + "/pdf")

    if host.endswith("arxiv.org") and parsed.path.startswith("/pdf/"):
        candidates.append(url)

    if host.endswith("sciencedirect.com") and "/science/article/pii/" in parsed.path:
        pii = (
            parsed.path.split("/science/article/pii/", 1)[1]
            .strip("/")
            .split("/", 1)[0]
        )
        candidates.append(
            "https://www.sciencedirect.com/science/article/pii/"
            f"{pii}/pdfft?isDTMRedir=true&download=true"
        )

    if host.endswith("ajog.org") and "/article/" in parsed.path:
        candidates.append(url.rstrip("/") + "/pdf")

    if host.endswith("lww.com") and "/fulltext/" in parsed.path:
        candidates.append(url.rstrip("/") + "/pdf")

    return list(dict.fromkeys(candidates))


def _doi_pdf_candidates(doi: str) -> list[str]:
    clean_doi = doi.strip()
    if not clean_doi:
        return []

    encoded_doi = quote(clean_doi, safe="/")
    lower = clean_doi.lower()
    candidates: list[str] = []
    if lower.startswith("10.1371/journal."):
        candidates.append(
            f"https://journals.plos.org/plosone/article/file?id={encoded_doi}&type=printable"
        )
    if lower.startswith("10.1186/"):
        candidates.append(f"https://link.springer.com/content/pdf/{encoded_doi}.pdf")
    if lower.startswith("10.1038/"):
        candidates.append(
            f"https://www.nature.com/articles/{clean_doi.split('/', 1)[1]}.pdf"
        )
    if lower.startswith("10.3389/"):
        candidates.append(f"https://www.frontiersin.org/articles/{encoded_doi}/pdf")
    return candidates


def _download_pdf_from_reference(reference: dict[str, object], timeout: int) -> bytes:
    pmcid = str(reference.get("pmcid") or "").strip().upper()
    if not pmcid:
        for url in reference.get("urls") or []:
            pmcid = _extract_pmcid_from_url(str(url))
            if pmcid:
                break
    if pmcid:
        try:
            return _download_pmc_pdf(pmcid, timeout)
        except FETCH_ERRORS:
            pass

    doi = str(reference.get("doi") or "").strip()
    errors: list[str] = []
    for candidate in _doi_pdf_candidates(doi):
        try:
            return _validate_pdf(_download(candidate, timeout))
        except FETCH_ERRORS as error:
            errors.append(f"{candidate}: {error}")
    for url in reference.get("urls") or []:
        for candidate in _candidate_pdf_urls(str(url), doi):
            try:
                return _validate_pdf(_download(candidate, timeout))
            except FETCH_ERRORS as error:
                errors.append(f"{candidate}: {error}")
    raise ValueError("; ".join(errors[-3:]) if errors else "no open PDF candidate")


def _source_key(reference: dict[str, object]) -> str:
    for field in ("pmcid", "doi"):
        value = str(reference.get(field) or "").strip().lower()
        if value:
            return f"{field}:{value}"
    urls = reference.get("urls") or []
    if urls:
        return f"url:{str(urls[0]).rstrip('/')}"
    return f"source:{reference.get('source_id')}"


def fetch_pmc_pdfs(
    manifest_path: str | Path,
    pdf_dir: str | Path,
    *,
    overwrite: bool = False,
    timeout: int = 60,
    limit: int | None = None,
) -> list[FetchResult]:
    """Download unique PMC PDFs listed in a manifest.

    Publisher/paywalled URLs are intentionally not fetched. The return value can
    be written as JSON to show which papers still need manual PDF upload.
    """

    references = load_manifest(manifest_path)
    output_dir = Path(pdf_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_pmcids: set[str] = set()
    results: list[FetchResult] = []

    for reference in references:
        pmcid = str(reference.get("pmcid") or "").strip().upper()
        if not pmcid or pmcid in seen_pmcids:
            continue
        seen_pmcids.add(pmcid)
        source_id = str(reference.get("source_id") or pmcid)
        destination = output_dir / f"{source_id}_{pmcid}.pdf"
        if destination.exists() and not overwrite:
            results.append(
                FetchResult(source_id, pmcid, str(destination), "skipped", "exists")
            )
            continue

        try:
            content = _download_pmc_pdf(pmcid, timeout)
            destination.write_bytes(content)
            results.append(FetchResult(source_id, pmcid, str(destination), "downloaded"))
        except FETCH_ERRORS as error:
            results.append(
                FetchResult(source_id, pmcid, str(destination), "failed", str(error))
            )

        if limit is not None and len(results) >= limit:
            break

    return results


def fetch_reference_pdfs(
    manifest_path: str | Path,
    pdf_dir: str | Path,
    *,
    overwrite: bool = False,
    timeout: int = 60,
    limit: int | None = None,
) -> list[FetchResult]:
    """Best-effort download of open PDFs from all manifest references."""

    references = load_manifest(manifest_path)
    output_dir = Path(pdf_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_keys: set[str] = set()
    results: list[FetchResult] = []

    for reference in references:
        source_id = str(reference.get("source_id") or "UNKNOWN")
        key = _source_key(reference)
        if key in seen_keys:
            results.append(FetchResult(source_id, "", "", "duplicate", key))
            continue
        seen_keys.add(key)

        pmcid = str(reference.get("pmcid") or "").strip().upper()
        destination = output_dir / f"{source_id}.pdf"
        if destination.exists() and not overwrite:
            results.append(
                FetchResult(source_id, pmcid, str(destination), "skipped", "exists")
            )
            continue

        try:
            content = _download_pdf_from_reference(reference, timeout)
            destination.write_bytes(content)
            results.append(FetchResult(source_id, pmcid, str(destination), "downloaded"))
        except FETCH_ERRORS as error:
            results.append(
                FetchResult(source_id, pmcid, str(destination), "failed", str(error))
            )

        attempted = sum(1 for result in results if result.status != "duplicate")
        if limit is not None and attempted >= limit:
            break

    return results


def write_fetch_report(results: list[FetchResult], report_path: str | Path) -> None:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([result.__dict__ for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
