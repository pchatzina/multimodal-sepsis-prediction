# Tests

Integration tests that validate pipeline outputs against a live PostgreSQL database.
No mocking — these require the MIMIC-IV database to be populated and the pipeline stages to have been run.

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific file
pytest tests/test_data_acquisition_pipeline.py -v

# Specific test
pytest tests/test_ehr_exports.py::TestCohortExport::test_no_non_cohort_subjects -v
```

## Test Files

| File | Validates | Requires |
|------|-----------|----------|
| `test_data_acquisition_pipeline.py` | Cohort integrity, modality links, raw files on disk | Phases 1–4 complete |
| `test_data_splitting.py` | Split completeness, no leakage, 70/15/15 ratios | Phase 5 complete |
| `test_ehr_exports.py` | EHR CSV exports match DB (cohort filter, test-split exclusion) | EHR export scripts run |

## Shared Fixtures

`conftest.py` provides `db_engine` (module-scoped SQLAlchemy engine) used by all test files.
