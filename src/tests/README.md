# IMI error-methods tests

Unit tests for the productionized inversion error methods (softplus solver, correlated prior-error
covariance Sa, correlated observational-error covariance So).

## Run

```bash
conda activate imi_env      # or any env with numpy/scipy/xarray/pyyaml/matplotlib/pytest
export MPLBACKEND=Agg
python -m pytest src/tests/ -v
```

`conftest.py` puts the repo root, `src/inversion_scripts`, and `src/utilities` on `sys.path`
(inversion_scripts before utilities, because both contain a `utils.py`).

## Coverage

| File | What it guards |
|------|----------------|
| `test_config_schema.py` | The `SoftplusErrors` / correlated-Sa / correlated-So keys are registered in the sanitizer as **optional** (type-checked, never required). |
| `test_softplus_dispatch.py` | `SoftplusErrors: true` selects the softplus solver; `InversionMethod: softplus` back-compat; the flag wins when both set. |
| `test_two_component_sa.py` | Two-component Sa cell-cell correlation = 1/(ratio·ratio); fully correlated at R01=1; Saunois global background applied. |
| `test_rem_diagonal.py` | REM diagonal So = Sk·g(P)/g(1): individual-observation variance at P=1, monotone reduction with super-ob count. |
