"""FastAPI application for the fetal brain MRI biometry calculator."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on the path when running from the app directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import Annotated

import jinja2
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from fetal_brain_mri.calculator import CaseResult, evaluate_case, evaluate_parameter
from fetal_brain_mri.gestational_age import parse_gestational_age
from fetal_brain_mri.inputs import MeasurementInput
from fetal_brain_mri.report import render_structured_report

app = FastAPI(title="Fetal Brain MRI Biometry Calculator")

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
