# PROGRESS.md

## Status

Project implementation has started from the specification-only repository.

## Completed

- Documentation seed files exist: `SPEC.md`, `TEST.md`, `AGENTS.md`, and `PLANS.md`.
- Added Python project scaffold with pytest, Ruff, and mypy configuration.
- Implemented gestational age parsing for `W+D`, `W w D d`, and decimal-week forms.
- Covered SPEC 4.1 patient-context entry and TEST 1.3 GA format requirements with tests.
- Implemented the three SPEC 4.2.1 normative model families and standard-normal percentile conversion.

## In Progress

- Increment 3: first source-result and consensus reconciliation slice from SPEC 4.2.3.

## Remaining

- FastAPI/Jinja/HTMX application scaffold.
- Normative source registry and model evaluators.
- Multi-source consensus engine.
- Differential diagnosis engine.
- Deterministic structured report generator.
- Clinical calculator UI.
- Conversion of `TEST.md` into machine-readable fixtures.
- Full validation, lint, and typecheck coverage for all SPEC acceptance criteria.
