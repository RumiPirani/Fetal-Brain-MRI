# PROGRESS.md

## Status

Project implementation has started from the specification-only repository.

## Completed

- Documentation seed files exist: `SPEC.md`, `TEST.md`, `AGENTS.md`, and `PLANS.md`.
- Added Python project scaffold with pytest, Ruff, and mypy configuration.
- Implemented gestational age parsing for `W+D`, `W w D d`, and decimal-week forms.
- Covered SPEC 4.1 patient-context entry and TEST 1.3 GA format requirements with tests.
- Implemented the three SPEC 4.2.1 normative model families and standard-normal percentile conversion.
- Implemented the first SPEC 4.2.3 consensus reconciliation core with source detail,
  in-range/extrapolated tagging, and `single`/`agree`/`disagree` states.
- Added the first SPEC 4.2.2 source-registry entries for TCD: Luis 2025 and Dovjak 2021.

## In Progress

- Increment 5: parameter bands from TEST 1.3 and SPEC 4.2 output semantics.

## Open Spec Issues

- SPEC 4.2.4's TCD worked example prose appears arithmetically inconsistent with its
  printed Luis coefficients. The encoded coefficients yield Luis mean 31.8764 mm,
  z = 0.817; paired with the Dovjak z = -0.165, the disagreement width is about
  0.982, which is `agree` under the SPEC 4.2.3 threshold. The prose says width
  1.003 and `disagree`. Current implementation follows the printed coefficients.

## Remaining

- FastAPI/Jinja/HTMX application scaffold.
- Normative source registry and model evaluators.
- Multi-source consensus engine.
- Differential diagnosis engine.
- Deterministic structured report generator.
- Clinical calculator UI.
- Conversion of `TEST.md` into machine-readable fixtures.
- Full validation, lint, and typecheck coverage for all SPEC acceptance criteria.
