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

