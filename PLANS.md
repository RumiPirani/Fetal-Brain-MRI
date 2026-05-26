# PLANS.md

Implementation plan for the Fetal Brain MRI Biometry Calculator.

## Current Inputs

- `SPEC.md` is the canonical standalone specification.
- `TEST.md` is the canonical test corpus.
- `Copy of Fetal Brain MRI Calculator - Design Doc.pdf` is the original design document.

## Phase 0: Project Scaffolding

- Choose the app stack and create the initial project structure.
- Add linting, formatting, unit test, and browser test commands.
- Establish a pure calculation package/module separate from the UI.
- Create structured registries for normative sources, DDx rules, report templates, and citations.
- Add CI-friendly commands for running all tests.

Exit criteria:

- The app starts locally.
- A placeholder calculator route exists.
- Tests can run from a clean checkout.

## Phase 1: Manual-Entry Calculator

### 1. Gestational Age and Input Model

- Implement gestational age parsing for `W+D`, `W d`, and decimal-week input.
- Validate supported GA ranges and mark extrapolation where source windows require it.
- Define the complete measurement input schema with units, optional qualitative fields, and missing-value handling.
- Add derived measurements needed by the spec, including asymmetry and summary size metrics.

### 2. Normative Source Registry

- Encode all Phase 1 normative sources from `SPEC.md` Section 7.
- Implement z-score and percentile calculations.
- Support quadratic, linear, percentile-derived, fixed-threshold, and source-specific model forms required by the spec.
- Add source metadata: citation, GA window, parameter coverage, notes, and validation caveats.

### 3. Multi-Source Consensus Engine

- Implement single-source pass-through.
- Implement multi-source consensus z-score/percentile calculation.
- Implement `single`, `agree`, and `disagree` states.
- Include per-source z-score disclosure in every result.
- Include extrapolation markers for out-of-window sources.
- Implement the source-registry admission check using the 0.5 SD maximum divergence rule.

### 4. Differential Diagnosis Engine

- Encode base triggers from `SPEC.md` Section 4.6.
- Encode combined-pattern cards, including ventriculomegaly, ACC, HPE, Dandy-Walker spectrum, aqueductal stenosis, posterior-fossa patterns, overgrowth patterns, and Chiari II/open neural tube defect.
- Add exclusion logic so negative-control patterns do not over-fire.
- Preserve citations, likelihoods, limitations, and recommended next steps.

### 5. Structured Report Output

- Generate deterministic plain-text report output.
- Include methodology text for multi-source consensus mode.
- Include findings, impression, source disclosure, DDx cards, and source-agreement notes.
- Add copy-to-clipboard support for PowerScribe workflow.
- Ensure no PHI fields are present.

### 6. Clinical UI

- Build the primary calculator interface.
- Add measurement input sections for supratentorial, ventricular, midline, posterior fossa, and brainstem measurements.
- Show immediate validation feedback, z-scores, percentiles, bands, and DDx cards.
- Add methodology/source views for formulas and citations.
- Add responsive desktop-first layout and browser accessibility checks.

Exit criteria:

- A user can manually enter all measurements, review results, and copy a structured report.
- All deterministic calculator logic is covered by unit tests.
- Core UI workflows pass browser verification.

## Phase 2: Test Corpus Conversion

- Convert `TEST.md` into machine-readable fixtures.
- Preserve case IDs, GA, measurements, expected bands, expected agreement states, fired DDx cards, non-fired DDx cards, impression text, and citations.
- Add unit tests for every case in the corpus.
- Add focused boundary tests for:
  - normal controls,
  - GA lower and upper limits,
  - severe ventriculomegaly threshold,
  - Dovjak/Luis source agreement,
  - extrapolated source markers,
  - multi-card simultaneous-fire reports,
  - negative-control combined-card subsumption.

Exit criteria:

- Every `TEST.md` fixture either passes or has a documented spec issue.
- Test output identifies the case ID and failed expectation.

## Phase 3: Validation and QI Support

- Add tools for exporting de-identified aggregate calculator results for QI analysis.
- Support the pre/post QI metrics specified in `SPEC.md`:
  - report completeness,
  - explicit z-score/percentile documentation,
  - reporting time,
  - recommendation or interpretation standardization.
- Add reproducible source cross-validation audit charts and status labels.
- Prepare FeTA and institutional cohort validation scripts if data becomes available.

Exit criteria:

- The app can support a QI study without storing PHI in the calculator itself.
- Source audit output is reproducible from the registry.

## Phase 4: Optional GenAI Report Drafting

This phase is optional and should not block the Phase 1 manual-entry calculator.

- Keep deterministic findings generation as the safety baseline.
- Use GenAI only for impression synthesis if explicitly requested.
- Ground all generated claims in a curated RAG knowledge bank or explicitly flagged PubMed search fallback.
- Add post-generation verification against the original numerical inputs.
- Fail closed to deterministic templates when verification detects a discrepancy.
- Keep backend privacy boundaries explicit and avoid PHI unless a compliant deployment plan exists.

Exit criteria:

- Generated report text cannot alter input measurements.
- Every non-template clinical claim is citation-grounded.
- Failed verification produces a safe deterministic report.

## Phase 5: Optional AI-Assisted Measurement Pre-Filling

This phase is future work and should follow successful manual calculator validation.

- Integrate or wrap an open-source fetal MRI biometry pipeline such as auto-proc-SVRTK.
- Accept 3-D SVR NIfTI inputs only in an environment designed for image data handling.
- Pre-fill calculator fields with AI measurements for radiologist review.
- Compare automated measurements against expert measurements and published inter-rater limits.
- Measure impact on reporting time and inter-observer variability.

Exit criteria:

- AI measurements are reviewable before use.
- Manual-entry workflow remains available.
- Validation demonstrates acceptable measurement agreement.

## Open Questions

- Final frontend/backend stack.
- Whether Phase 1 includes all high-yield added parameters or launches with the core fourteen inputs first.
- Exact institutional requirements for Epic Radiant launch and PowerScribe workflow.
- Whether source data should be hand-entered into TypeScript/JSON, generated from a spreadsheet, or maintained in a separate clinical registry file.
- Which validation cohort data will be available during implementation.

