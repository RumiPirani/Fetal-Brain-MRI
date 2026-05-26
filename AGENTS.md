# AGENTS.md

Guidance for coding agents working on the Fetal Brain MRI Biometry Calculator.

## Source of Truth

- Treat `SPEC.md` as the canonical product and clinical specification.
- Treat `TEST.md` as the canonical validation corpus.
- The PDF design document is the originating design artifact, but the Markdown files are easier to search and should be used for implementation details.
- Do not change clinical formulas, thresholds, diagnostic likelihoods, report language, source labels, or citation claims unless the change is explicitly requested and reflected in the spec.

## Project Goal

Build a browser-based, workflow-integrated fetal brain MRI biometry calculator that:

- accepts gestational age and fetal brain MRI measurements,
- computes z-scores, percentiles, and normal/abnormal bands from literature-based reference curves,
- reconciles multiple normative sources with transparent source disclosure,
- fires evidence-based differential diagnosis cards,
- emits structured report text that can be copied into PowerScribe,
- runs without collecting, transmitting, or storing PHI.

The Phase 1 product is a manual-entry calculator. AI-assisted measurement pre-filling and GenAI report drafting are future phases unless explicitly requested.

## Clinical Safety Rules

- This is clinical decision support, not autonomous diagnosis.
- Keep all abnormal outputs traceable to deterministic measurements, thresholds, and citations.
- Never invent medical claims, references, probabilities, or next-step recommendations.
- Show per-source z-scores and source labels for each measurement used in a consensus result.
- Flag source disagreement when in-range source z-scores differ by at least 1.0 SD.
- Flag extrapolated sources when gestational age falls outside a source's validated GA window.
- If a value is missing, invalid, or out of supported range, fail visibly and conservatively.
- Avoid patient identifiers entirely. The app should not ask for name, MRN, accession, DOB, or other PHI.

## Required Calculator Inputs

Core manual-entry inputs are:

- gestational age, accepted as completed weeks plus days or decimal weeks,
- skull BPD,
- skull OFD,
- brain BPD,
- brain OFD left,
- brain OFD right,
- right atrial diameter,
- left atrial diameter,
- cavum septum pellucidum width and presence/absence status,
- corpus callosum length and presence/absence status,
- transcerebellar diameter,
- vermis cranio-caudal height,
- vermis antero-posterior diameter,
- pons antero-posterior diameter,
- third ventricle width.

The design also calls for high-yield Phase 1 additions:

- cisterna magna depth,
- tegmento-vermian angle,
- maximum transverse diameter of the posterior fossa,
- clivus-supraocciput angle.

## Normative Engine Requirements

- Represent each normative source as structured data with parameter, source label, valid GA range, mean model, standard deviation or percentile model, citation, and notes.
- Compute each source-specific z-score as specified in `SPEC.md`.
- Convert z-scores to percentiles using the standard normal CDF.
- For parameters with one applicable source, pass through that source result and mark agreement as `single`.
- For parameters with multiple applicable sources, compute the consensus value from contributing sources and mark agreement as:
  - `agree` when source z-scores differ by less than 1.0 SD,
  - `disagree` when source z-scores differ by at least 1.0 SD.
- Keep extrapolated results visible. Do not silently drop or hide them.
- Derived summary measures such as head circumference or asymmetry should be computed deterministically and covered by tests.

## Differential Diagnosis Engine

- Implement DDx cards as deterministic rules over measurement bands, z-scores, qualitative fields, and combined patterns.
- Preserve the distinction between base triggers and combined-pattern cards.
- Ensure combined cards do not accidentally fire when exclusion criteria are present.
- Every diagnosis emitted by the engine must be supported by at least five cases in `TEST.md` unless the spec says otherwise.
- DDx output should include diagnosis name, trigger rationale, estimated likelihood where specified, limitations, recommended next steps, and citations.

## Report Generation

The deterministic structured report must include:

- every measured parameter,
- consensus z-score and percentile,
- every contributing source's individual z-score and source label,
- agreement state,
- extrapolated markers where applicable,
- abnormal findings and fired DDx cards,
- a fixed methodology statement describing multi-source consensus mode,
- a `SOURCE-AGREEMENT NOTES` block whenever any parameter is in `disagree`.

PowerScribe integration in Phase 1 is plain-text copy-to-clipboard only. Do not emit HTML, rich text, or PHI-bearing fields.

## Architecture Preferences

- Prefer a fully client-side application for Phase 1.
- Keep clinical calculation logic in a pure, testable module independent of UI state.
- Keep normative source data and DDx rules in auditable structured registries.
- Make report generation deterministic and testable.
- Do not require a backend for the Phase 1 calculator.
- If a future backend is introduced for RAG, PubMed search, AI pre-filling, or model inference, isolate it from the PHI-free Phase 1 calculator and document the privacy boundary.

## UI Expectations

- The first screen should be the actual calculator, not a marketing landing page.
- Favor dense, quiet clinical workflow UI over decorative layouts.
- Measurement inputs should show units, validation state, and concise measurement guidance.
- Abnormal results should be scannable and tied to the relevant measurement.
- Include copy-to-clipboard controls for report text.
- Include methodology/source transparency where users can inspect formulas, source windows, and citations.
- Ensure text and controls work on both desktop and mobile, but optimize primarily for clinical desktop use.

## Testing Expectations

- Build tests directly from `TEST.md`.
- Cover normal controls, every abnormal DDx section, negative controls, boundary conditions, GA extremes, multi-card simultaneous-fire cases, and multi-source agreement/disagreement behavior.
- Include exact expected band classifications and fired/non-fired DDx cards.
- Test gestational age parsing for both `W+D` and decimal-week forms.
- Test report text for required sections, source disclosure, disagreement notes, and no unexpected PHI fields.
- Any implementation of new source-registry entries must include the 0.5 SD source-admission check described in `SPEC.md`.

## Documentation

- Keep formulas and clinical logic documented close to their implementation.
- Link implementation constants back to the relevant `SPEC.md` section or source registry entry.
- If a spec ambiguity is discovered, document it as an open question instead of guessing silently.
- Keep this file updated when project conventions become concrete.

