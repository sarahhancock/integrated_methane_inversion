# National Inventory (BTR) Prior Covariance

This option builds `prior_norm_error_covariance.npz` from **per-(country, sector) inventory
uncertainties** ("bottom-up reported", BTR) instead of a single spatial correlation length scale. It
is used by both `PriorCovarianceMethod: national_inventory` and `PriorCovarianceMethod: sector_ensemble`
(the anthropogenic block of the sector ensemble is this two-component national covariance).

## Quick start — any IMI user, no extra files

A global admin-0 country shapefile (with `NAME` and `ISO3` columns) is **bundled with the IMI**
(`resources/countries/imi_country_boundaries.*`) and used by default, so you only need to supply your
uncertainty CSV. Match your countries by ISO3 code (recommended — names vary between inventories):

```yaml
OffDiagonalPriorCov: true
PriorCovarianceMethod: national_inventory
NationalPriorUncertaintyFile: /path/to/national_prior_uncertainties.csv
NationalPriorCountryNameColumn: ISO3        # ISO3 (recommended) or NAME; must match your CSV's countries
```

The CSV needs a country identifier, a `sector`, and a relative uncertainty. Columns are flexible:

```csv
iso3,sector,relative_uncertainty
COL,Oil,0.30
COL,Gas,0.30
COL,Coal,0.50
COL,Livestock,0.40
USA,Oil,0.25
USA,Gas,0.25
```

- **Country identifier column:** `iso3` | `country` | `country_name` | `country_id`. Its values must
  match the shapefile column named by `NationalPriorCountryNameColumn` (`ISO3` for ISO3 codes, `NAME`
  for country names). Header casing/whitespace is tolerated.
- **Uncertainty column:** `relative_uncertainty` | `u` | `uncertainty`. This is a **relative** 1-sigma
  (coefficient of variation), e.g. `0.30` = 30%. Blank or non-numeric cells are skipped with a note.
- **`sector`** must match the HEMCO prior emission field suffix after `EmisCH4_` (e.g. `EmisCH4_Oil` →
  `Oil`). Run the builder once and check the console/diagnostics for the sector names it found.

## Partial coverage and defaults

- **No `NationalPriorUncertaintyFile` at all:** the builder falls back to global sectoral uncertainties
  from Saunois et al. and treats the whole domain as one region (cells of the same sector correlated
  domain-wide). No country matching is done.
- **A (country, sector) present in the emissions but absent from your CSV:** that group is not given a
  BTR aggregate; its elements fall back to the uniform `PriorError` prior. So you can supply
  uncertainties for only the countries/sectors you care about; the rest keep the default.
- **A country in your CSV not found in the shapefile column:** it is skipped with a printed warning
  (`Country mask: Country 'XXX' not found ...; skipping`). If *no* group matches at all, the builder
  raises an error naming the likely cause (wrong `NationalPriorCountryNameColumn` or sector names).

## Advanced options

Use your own country boundaries instead of the bundle, or a precomputed gridded country mask:

```yaml
# your own shapefile
NationalPriorCountryShapefile: /path/to/countries.shp
NationalPriorCountryNameColumn: NAME
# OR a precomputed integer country mask on the GEOS-Chem grid, matched to a numeric country_id column
NationalPriorCountryMaskFile: /path/to/country_mask_on_gc_grid.nc
NationalPriorCountryMaskVariable: country_id
```

Other tuning keys: `NationalPriorGridNationalRatio` (grid:national error ratio R01, default 2.5),
`NationalPriorMinUncertainty` (floor on the reported uncertainty, default 0.30),
`NationalPriorTwoComponent` (national rank-1 + local diagonal; the sector-ensemble anthro block uses
this), `NationalPriorGlobalBackground` (Saunois floor, default on),
`NationalPriorDomainInvariant` (scale the national term by 1/f_C^2 for countries only partly in the
domain, using equal-area in-domain fractions).

## What it does / outputs

The builder uses `hemco_prior_emis` output to compute, for each total-methane scale-factor element,
how much of that element's prior emissions come from each country-sector pair. It then sets the
within-country-sector covariance so the emission-weighted **national aggregate uncertainty equals the
reported BTR value** (`max(u_BTR, Saunois floor)`), combines sectors by emission weight, repairs the
matrix to be positive semidefinite, and writes:

- `prior_norm_error_covariance.npz`  — the correlation + per-element `sigma_scale` invert.py consumes
- `national_inventory_prior_covariance_diagnostics.csv`  — per group: target, achieved, and floored
  relative uncertainties (a quick check that "achieved / u_BTR" is ~1.0 for every group)
