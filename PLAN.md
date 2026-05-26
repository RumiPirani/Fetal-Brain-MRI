# PLAN.md

## 2026-05-25 Increment 1: Gestational Age Scaffold

- Create the missing mission-tracking files expected by the work loop.
- Add a minimal Python project scaffold using `src/` and `tests/`.
- Write the first failing test for SPEC 4.1 and TEST 1.3 gestational age input formats.
- Implement only the gestational age parser and validation needed to pass that test.
- Run tests, linter, and typechecker; fix any failures before committing.

## 2026-05-25 Increment 2: Normative Model Families

- Add tests for the three SPEC 4.2.1 model families before implementation.
- Cover z-score and standard-normal percentile conversion without introducing SciPy yet.
- Implement immutable model classes with `mean`, `sigma`, `z_score`, and `percentile`.
- Validate that model sigma values are positive and fail clearly when not.
- Run tests, linter, and typechecker; fix any failures before committing.

## 2026-05-25 Increment 3: Consensus Reconciliation Core

- Add tests for SPEC 4.2.3 source evaluation and agreement states.
- Cover `single`, `agree`, `disagree`, and all-extrapolated fallback behavior.
- Implement source metadata, per-source results, and consensus result objects.
- Keep report and UI surfacing out of scope for this increment.
- Run tests, linter, and typechecker; fix any failures before committing.

## 2026-05-25 Increment 4: First Source Registry Entries

- Add tests for retrieving the SPEC 4.2.2/4.2.4 TCD registry entries.
- Encode Luis 2025 and Dovjak 2021 TCD model coefficients from SPEC 4.2.4.
- Verify per-source means, sigmas, z-scores, and consensus output.
- Record the SPEC 4.2.4 arithmetic discrepancy in `PROGRESS.md`.
- Run tests, linter, and typechecker; fix any failures before committing.

## 2026-05-25 Increment 5: Band Classification

- Add tests for TEST 1.3 band labels from consensus z-scores.
- Implement `<5th`, `normal`, and `>95th` labels using standard z cutoffs.
- Keep size-summary `<3rd` and `>97th` special cases out of scope until those
  derived summary parameters are implemented.
- Run tests, linter, and typechecker; fix any failures before committing.

## 2026-05-25 Increment 6: Parameter Evaluation API

- Add tests for a calculator-facing parameter evaluation function.
- Combine source lookup, consensus reconciliation, and standard band classification.
- Return a stable result object containing parameter id, measurement, consensus, and band.
- Fail clearly for unknown parameters via the registry lookup.
- Run tests, linter, and typechecker; fix any failures before committing.

