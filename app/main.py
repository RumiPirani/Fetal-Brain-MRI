"""FastAPI application for the fetal brain MRI biometry calculator."""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Ensure src is on the path when running from the app directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import Annotated

import jinja2
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from fetal_brain_mri.calculator import CaseResult, evaluate_case, evaluate_parameter
from fetal_brain_mri.gestational_age import parse_gestational_age
from fetal_brain_mri.inputs import MeasurementInput
from fetal_brain_mri.report import render_structured_report
from fetal_brain_mri.text_parser import parse_text
from fetal_rag.config import load_env_file
from fetal_rag.rag import TfidfRagEngine

app = FastAPI(title="Fetal Brain MRI Biometry Calculator")

# ---------------------------------------------------------------------------
# RAG engine (lazy-loaded on first /chat request)
# ---------------------------------------------------------------------------

_rag_engine: TfidfRagEngine | None = None


def _get_rag_engine() -> TfidfRagEngine:
    global _rag_engine
    if _rag_engine is None:
        load_env_file(Path(__file__).parent.parent / ".env.local")
        index_dir = Path(__file__).parent.parent / "vector_db" / "fetal_mri"
        _rag_engine = TfidfRagEngine.from_index(str(index_dir))
    return _rag_engine


# Use a pre-built Jinja2 environment with caching disabled to avoid a Python
# 3.14 / Jinja2 LRU-cache hashability incompatibility.
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(Path(__file__).parent / "templates")),
    autoescape=True,
    cache_size=0,
)
templates = Jinja2Templates(env=_jinja_env)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PARAM_FIELDS = [
    # (field_name, label, unit, section)
    ("skull_bpd",      "Skull BPD",                   "mm", "Skull"),
    ("skull_ofd",      "Skull OFD",                   "mm", "Skull"),
    ("brain_bpd",      "Brain BPD",                   "mm", "Brain parenchyma"),
    ("brain_ofd_l",    "Brain OFD (left)",             "mm", "Brain parenchyma"),
    ("brain_ofd_r",    "Brain OFD (right)",            "mm", "Brain parenchyma"),
    ("atrium_r",       "Atrial diameter (right)",      "mm", "Ventricles"),
    ("atrium_l",       "Atrial diameter (left)",       "mm", "Ventricles"),
    ("csp",            "CSP width",                   "mm", "Midline"),
    ("cc_length",      "Corpus callosum length",       "mm", "Midline"),
    ("tcd",            "Transcerebellar diameter",     "mm", "Posterior fossa"),
    ("vermis_cc",      "Vermis height (CC)",           "mm", "Posterior fossa"),
    ("vermis_ap",      "Vermis AP diameter",           "mm", "Posterior fossa"),
    ("tva_degrees",    "Tegmento-vermian angle",       "°",  "Posterior fossa"),
    ("cisterna_magna_depth", "Cisterna magna depth",  "mm", "Posterior fossa"),
    ("pons_ap",        "Pons AP diameter",             "mm", "Brainstem"),
    ("third_ventricle","Third ventricle width",        "mm", "Other"),
    ("tdpf",           "TDPF (posterior fossa diam)", "mm", "Chiari II"),
    ("csa",            "Clivus-supraocciput angle",   "°",  "Chiari II"),
]

_NON_Z_PARAMS = {"tva_degrees", "cisterna_magna_depth"}  # qualitative, not registered


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _build_inputs_from_form(form: dict[str, str]) -> MeasurementInput | None:
    ga_raw = form.get("ga", "").strip()
    if not ga_raw:
        return None
    try:
        ga = parse_gestational_age(ga_raw)
    except ValueError:
        return None

    kwargs: dict = {"ga_weeks": ga.decimal_weeks}
    for field_name, *_ in _PARAM_FIELDS:
        if field_name in _NON_Z_PARAMS:
            kwargs[field_name] = _parse_float(form.get(field_name))
        else:
            kwargs[field_name] = _parse_float(form.get(field_name))

    kwargs["csp_absent"] = form.get("csp_absent") == "on"
    kwargs["cc_absent"] = form.get("cc_absent") == "on"
    return MeasurementInput(**kwargs)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def calculator_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "calculator.html",
        {
            "param_fields": _PARAM_FIELDS,
            "non_z_params": _NON_Z_PARAMS,
            "case": None,
            "report_text": "",
            "error": None,
        },
    )


@app.post("/calculate", response_class=HTMLResponse)
async def calculate(request: Request) -> HTMLResponse:
    form_data = await request.form()
    form = dict(form_data)

    inputs = _build_inputs_from_form(form)
    error = None
    case: CaseResult | None = None
    report_text = ""

    if inputs is None:
        error = "Please enter a valid gestational age (e.g. 28+3 or 28.4)."
    else:
        try:
            case = evaluate_case(inputs)
            report_text = render_structured_report(
                list(case.parameters.values()), case=case
            )
        except Exception as exc:  # noqa: BLE001
            error = f"Calculation error: {exc}"

    return templates.TemplateResponse(
        request,
        "calculator.html",
        {
            "param_fields": _PARAM_FIELDS,
            "non_z_params": _NON_Z_PARAMS,
            "case": case,
            "report_text": report_text,
            "error": error,
            "form": form,
        },
    )


@app.post("/parse-text", response_class=JSONResponse)
async def parse_text_endpoint(request: Request) -> JSONResponse:
    form_data = await request.form()
    text = str(form_data.get("text", "")).strip()
    if not text:
        return JSONResponse({"values": {}, "unrecognized": [], "warnings": ["No text provided."]})
    result = parse_text(text)
    return JSONResponse({
        "values": result.values,
        "unrecognized": result.unrecognized,
        "warnings": result.warnings,
    })


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request) -> HTMLResponse:
    form_data = await request.form()
    question = str(form_data.get("question", "")).strip()
    report_context = str(form_data.get("report_context", "")).strip() or None

    if not question:
        return HTMLResponse("")

    try:
        engine = _get_rag_engine()
        result = engine.answer(question, calculator_data=report_context, top_k=6)
        answer_text = result.answer
        citations = result.contexts
        error_msg = None
    except Exception as exc:  # noqa: BLE001
        answer_text = ""
        citations = []
        error_msg = str(exc)

    q = html.escape(question)

    if error_msg:
        body_html = f'<p class="text-red-600 text-xs">{html.escape(error_msg)}</p>'
    else:
        a = html.escape(answer_text).replace("\n", "<br>")
        body_html = f'<p class="text-sm text-gray-800">{a}</p>'
        if citations:
            items = "".join(
                f'<div class="mb-1"><span class="font-semibold text-gray-700">[{c.rank}]</span> '
                f'{html.escape(c.chunk.source_title or c.chunk.source_id)}'
                f'{f" p.{c.chunk.page_start}" if c.chunk.page_start else ""} — '
                f'<span class="text-gray-500">{html.escape(c.chunk.text[:220])}…</span></div>'
                for c in citations
            )
            body_html += (
                f'<details class="mt-2">'
                f'<summary class="cursor-pointer text-xs text-blue-600 select-none">'
                f'{len(citations)} source(s)</summary>'
                f'<div class="mt-1 text-xs text-gray-600 space-y-1 pl-1">{items}</div>'
                f'</details>'
            )

    fragment = (
        f'<div class="mb-4">'
        f'<div class="mb-1 text-xs font-semibold text-blue-700">You</div>'
        f'<div class="bg-blue-50 rounded px-3 py-2 text-sm text-gray-900 mb-2">{q}</div>'
        f'<div class="text-xs font-semibold text-gray-500 mb-1">AI — literature-grounded</div>'
        f'<div class="bg-gray-50 border border-gray-200 rounded px-3 py-2">{body_html}</div>'
        f'</div>'
    )
    return HTMLResponse(fragment)
