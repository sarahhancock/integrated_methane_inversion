# National Inventory Prior Covariance

This option builds `prior_norm_error_covariance.npz` for total-methane scale
factor inversions using sector inventory uncertainty constraints instead of a
spatial correlation length scale.

## Global defaults (Saunois et al.)

When no `NationalPriorUncertaintyFile` is provided, the builder uses global
sectoral uncertainties from Saunois et al. and treats the entire inversion
domain as one region. Grid cells dominated by the same emission sector are
correlated across the whole domain.

```yaml
OffDiagonalPriorCov: true
PriorCovarianceMethod: national_inventory
```

## Country-level uncertainties

To apply different uncertainties per country, provide an uncertainty CSV and a
country mask:

```yaml
OffDiagonalPriorCov: true
PriorCovarianceMethod: national_inventory
NationalPriorUncertaintyFile: /path/to/national_prior_uncertainties.csv
NationalPriorCountryMaskFile: /path/to/country_mask_on_gc_grid.nc
NationalPriorCountryMaskVariable: country_id
```

or use the same kind of countries shapefile used in project utility code:

```yaml
OffDiagonalPriorCov: true
PriorCovarianceMethod: national_inventory
NationalPriorUncertaintyFile: /path/to/national_prior_uncertainties.csv
NationalPriorCountryShapefile: /path/to/countries.shp
NationalPriorCountryNameColumn: NAME
NationalPriorCountryMaskAreaWeighting: true
```

The uncertainty CSV must contain:

```csv
country_id,country,sector,relative_uncertainty
170,Colombia,Oil,0.30
170,Colombia,Gas,0.30
```

`sector` should match the HEMCO prior emission field suffix after `EmisCH4_`.
For example, `EmisCH4_Oil` is written as `Oil`.

When using a shapefile, `country` is matched against
`NationalPriorCountryNameColumn`. `country_id` may be left blank, in which case
the builder assigns sequential internal IDs.

The builder uses `hemco_prior_emis` output to compute, for each total methane
scale-factor element, how much of that element's prior emissions come from each
country-sector pair. It then estimates within-country-sector correlations needed
to reproduce the reported aggregate uncertainty, combines those correlations
across sectors by emission weights, repairs the resulting matrix to be positive
semidefinite, and writes:

- `prior_norm_error_covariance.npz`
- `national_inventory_prior_covariance_diagnostics.csv`

The diagnostics CSV reports the raw and clipped correlation for each
country-sector group plus target, independent, fully correlated, and achieved
relative uncertainties.
