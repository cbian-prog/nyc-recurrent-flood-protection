# Flood protection built for rare losses misses the floods that recur

Code and data for Bian, C., Horton, R., Bresch, D. N. & Zheng, T. *Flood protection built for rare losses misses the floods that recur* (submitted to *Nature Cities*, 2026).

The study places New York City's flood sources (coastal surge, rainfall and tide), the assets they reach (buildings, basement homes and subway stations) and three protection instruments (the FEMA flood map, the sewer design standard and the National Flood Insurance Program) on one frequency axis. Every input is a public dataset.

- Code: this repository, archived at Zenodo, https://doi.org/10.5281/zenodo.23962188
- Large public inputs as retrieved for the study: Zenodo, https://doi.org/10.5281/zenodo.23962247
- Licences: code MIT (`LICENSE`); data produced by this study CC BY 4.0 (`LICENSE-DATA`); third-party inputs keep their original terms (see *Inputs* below)

## Contents

| Path | What it is |
|---|---|
| `*.py` | Analysis and figure scripts (listed below). Run from the repository root. |
| `*.json` | Results written by the analysis scripts and read by the figure and table scripts. |
| `figures/` | Figures 1–4, Extended Data Figs 1–4 and Supplementary Figs 1–3 as published (PDF and PNG). |
| `tables/` | Supplementary Tables 1–7 (CSV and one workbook) and their legends. |
| `floodnet_cleaned/` | The cleaned FloodNet street-flood record used in the paper, with every excluded event and its reason (own README, CC BY 4.0). |
| `data/paper_a/` | Intermediate products of this study (described below). |
| `data/naccs/` | NACCS still-water curve at The Battery digitised for this study, its tropical and extratropical components, and the per-save-point reconstruction. |
| `data/` (other) | Small public inputs as retrieved (snapshots of sources that change over time). |
| `MANIFEST.sha256` | SHA-256 and size of every file. |

## Quick start: redraw the figures and tables

The results needed by the figure scripts are included, so the figures and tables can be redrawn without rerunning the analyses. Most figure scripts also read map layers from the large-input archive, so unpack it first (see *Inputs*).

```bash
conda env create -f environment.yml
conda activate nyc-recurrent-floods
# unpack the Zenodo input archive https://doi.org/10.5281/zenodo.23962247 into this folder, so that its data/ merges with ./data/
python nc_fig1.py; python nc_fig2.py; python nc_fig3.py; python nc_fig4_ed78.py
python nc_edfig_record.py; python nc_edfig_surge.py; python nc_ed456.py
python nc_si_robust.py; python nc_si_placement.py; python nc_supp.py
python make_supp_tables.py
```

Figures are written to `figures/` and tables to `tables/`.

## Full reproduction

`run_all.sh` runs every analysis in dependency order and then draws all figures and tables. It assumes the input archive has been unpacked. The storm-and-place model (`nc_storm_model.py`, 100 storm-day bootstrap refits) and the Monte Carlo (`paper_a_mc.py`, 4,000 draws) take the longest; everything else runs in minutes on a laptop.

| Stage | Script | Paper item | Output |
|---|---|---|---|
| Surge model | `naccs_aai.py` | Methods (hazard curve); ED Fig. 2d | `naccs_aai.json` |
| | `naccs_jpm_points.py` | ED Fig. 2e,f | `data/naccs/chs/jpm_points.json` (included; rerunning needs the CHS files, see *Inputs*) |
| | `paper_a_core.py` | Surge loss, Sandy check | `paper_a_core.json`, `data/paper_a/depths.npz` |
| | `paper_a_mc.py` | AAI uncertainty; ED Fig. 2c; Suppl. Table 4 | `paper_a_mc.json` |
| | `figED1_validation.py` | ED Fig. 2a–c | `figED1_validation.json` |
| | `figED5_vulnerability.py` | ED Fig. 2g–i | `figED5_vulnerability.json` |
| | `nc_valuation_sensitivity.py` | Suppl. Methods (valuation) | `nc_valuation_sensitivity.json` |
| Areas | `nc_areas.py`, `nc_citygrid.py` | Footprint areas, sensor siting; lookup grid | `nc_areas.json` |
| Street floods | `nc_fog_check.py` | Rain evidence for each event | `data/paper_a/event_rain_class.csv` |
| | `nc_tidal_reclass.py` | Three sensors treated as tidal | `data/paper_a/tidal_reclass.json` |
| | `nc_events.py` | Flood definition and cleaning (module) | — |
| | `nc_data.py` | Numbers in Results; Fig. 1c, Fig. 3 | `nc_numbers.json`, `data/paper_a/*` |
| | `nc_si_transfer.py` | Borough transfer; Suppl. Fig. 1h | `nc_si_transfer.json`, `data/paper_a/sensor_days.parquet` |
| | `nc_si_subhourly.py` | Sub-hourly rain; Suppl. Fig. 1e–g; Suppl. Note 2 | `nc_si_subhourly.json` |
| | `nc_si_duration.py` | 30-min rule; Suppl. Fig. 1a–d | `nc_si_duration.json` |
| | `nc_storm_stats.py` | Storm concentration, storm-day intervals; ED Fig. 1b | `nc_storm_stats.json` |
| | `nc_rain_climatology.py` | Long-run rescaling; Suppl. Fig. 1i | `nc_rain_climatology.json` |
| | `nc_storm_model.py` | Storm-and-place model; ED Fig. 1c,d; Suppl. Note 1 | `nc_storm_model.json` |
| | `nc_tide_effect.py` | High-tide effect; ED Fig. 1e | `nc_tide_effect.json` |
| | `nc_package_floods.py` | Released flood record | `floodnet_cleaned/` |
| Exposure | `basement_exposure.py` | Basement lots in each footprint | `basement_summary.json` |
| | `figED6_basements.py` | Informal units; ED Fig. 3a; Suppl. Table 1 | `figED6_basements.json` |
| | `subway_flood.py` | Stations and entrances | `subway_summary.json` |
| | `slr_overlay.py` | Land inundated at +3 ft | `slr_summary.json` |
| | `fig2_impact_rp.py` | Impact against return period; Fig. 2b,c | `fig2_impact_rp.json` |
| Insurance | `fig5_gap_nsi.py` | Insurable shares and decomposition; Fig. 4; ED Fig. 4d,e | `fig5_gap_nsi.json` |
| | `figED7_nfip.py` | Policy file, Sandy claims, block-group map; ED Fig. 4a–c | `figED7_nfip.json` |
| | `nfip_sandy_claims_v3.py` | Sandy claims (downloads from OpenFEMA) | `nfip_sandy_claims_v3.json` |
| | `pluvial_insurance_nsi.py` | Rain-side and surge-side insurance; Fig. 3b; Suppl. Table 6 | `pluvial_insurance_nsi.json` |
| | `insurance_city.py` (`CITY=nyc`, `CITY=miami`) | Two-city comparison; Suppl. Fig. 3a; Suppl. Table 7 | `insurance_city_*.json` |
| | `figED3_hazard_basis.py` | Category against water-level basis; Suppl. Fig. 3b | `figED3_hazard_basis.json` |
| | `download_pfirm.py` | 2015 preliminary flood map (downloads from FEMA) | `data/fema_nfhl/pfirm2015_zones_nyc.geojson` |
| | `nc_si_sensitivity.py` | Suppl. Table 2 | `nc_si_sensitivity.json`, `tables/Table_S2.csv` |
| Figures | `nc_fig1.py`, `nc_fig2.py`, `nc_fig3.py`, `nc_fig4_ed78.py` | Figs 1–4 and ED Fig. 4 | `figures/` |
| | `nc_edfig_record.py`, `nc_edfig_surge.py`, `nc_ed456.py` | ED Figs 1–3 | `figures/` |
| | `nc_si_robust.py`, `nc_si_placement.py`, `nc_supp.py` | Suppl. Figs 1–3 | `figures/` |
| Tables | `make_supp_tables.py` | Suppl. Tables 1–7 | `tables/` |
| Style | `nc_style.py` | Shared figure style | — |

Several scripts carry names from earlier drafts (`figED*`, `fig5_*`); the table above gives the item each one now supports. Files whose name contains `ED`, `fig` or `nc_` are all part of the published analysis; nothing else is needed.

## Data produced by this study

| File | Content |
|---|---|
| `floodnet_cleaned/` | Every FloodNet event in the source table, with the 30-min drain rule, the rain-evidence test and the tidal flag applied (the paper analyses events to 11 August 2026); excluded events with reason codes (`SHORT`, `NO_RAIN`, `SHORT+NO_RAIN`, `NO_SENSOR`); sensor table; rainy sensor-days. See its README. |
| `data/paper_a/depths.npz` | Depth above ground at each of the 881,621 NSI structures (`depths`, 29 × 881,621) for 29 still-water levels from 0.5 to 7.5 m NAVD88 (`levels`); connected bathtub on the 20 m grid. |
| `data/paper_a/masks_20m.npz`, `masks_city.npz` | Boolean rasters on the 20 m grid, for the surge window and for the whole city: FEMA Special Flood Hazard Area (`sfha`), 1-in-100 and 1-in-1,000-year surge reach (`s100`, `s1000`), extreme rainfall footprint (`rain`) and city land (`city`). |
| `data/paper_a/sensor_days.parquet`, `sensor_days_model.parquet` | One row per rainy day at each non-tidal sensor: flood (`y`) and deepest flood depth (`dmax`), radar maximum hourly rain (`pmax`), Battery level relative to MHHW and high-tide flag (`tide_mhhw`, `hi`), distance to open water (`to_water`, `near`); the first file adds peak hour, intensity bin, borough and coordinates. |
| `data/paper_a/lot_rain_bins.npz` | Rainy days in each of four intensity bins over the radar record (`nl`, 4 × 22,443) at the radar cell of each basement lot in the extreme rainfall footprint, with coordinates, distance to water and record length (`span_yr`). |
| `data/paper_a/event_rain_class.csv`, `tidal_reclass.json` | Rain evidence for each event; the three sensors reclassified as tidal and the rule. |
| `data/mrms/*.npz` | NOAA MRMS MultiSensor QPE 1-h Pass 2, cropped to New York City, one file per UTC day (1,172 days). |
| `data/naccs/battery_naccs_curve.json` | Mean NACCS still-water curve at save point 7672 (The Battery), digitised from TR-15-5 Fig. 8-19 (±0.1 m), MSL and NAVD88. |
| `data/naccs/battery_naccs_tc_xc_curves.json` | Tropical and extratropical components of the same curve. |
| `data/naccs/chs/jpm_points.json` | Joint-probability curves reconstructed at save points 7672, 11982, 12049 and 14200. |
| `data/atlas14_centralpark_intensity_pds.csv` | NOAA Atlas 14 vol. 10 partial-duration intensities at Central Park, as exported from the PFDS. |
| `data/battery_annual_max.npy` | Annual maximum verified water level at The Battery, 1920–2025 (m NAVD88). |
| `data/nfip/tract_equity.parquet` | By census tract: LMI share, population, building value, NFIP contracts and building cover, take-up. |

## Inputs

All inputs are public. Small inputs that change over time are included in `data/` as retrieved; larger ones are in the Zenodo input archive (https://doi.org/10.5281/zenodo.23962247), which mirrors the `data/` layout. The NACCS per-save-point files require free registration at the USACE Coastal Hazards System and are not redistributed; the curves derived from them are in `data/naccs/`.

| Dataset | Provider and identifier | Accessed | Location |
|---|---|---|---|
| NACCS still-water return periods, storm rates and peaks | USACE Coastal Hazards System, https://chs.erdc.dren.mil (registration); report ERDC/CHL TR-15-5 | 2026 | derived curves in `data/naccs/` |
| Continuously Updated DEM, 1/9 arc-second | NOAA NCEI, https://doi.org/10.25921/ds9v-ky35 | 2026 | archive: `data/manhattan_dem_20m.tif`, `data/nyc_dem_55m.tif` (resampled) |
| Datums and verified water levels, station 8518750 | NOAA CO-OPS, https://tidesandcurrents.noaa.gov/stationhome.html?id=8518750 | 2026 | `data/coops/`, `data/battery_annual_max.npy` |
| Sea Level Rise inundation extent, MHHW + 0–6 ft | NOAA Office for Coastal Management, InPort 48106, https://coast.noaa.gov/slrdata | 2026 | `data/slr/` (resampled to 55 m) |
| SLOSH Maximum of Maximums v4 | NOAA NHC, https://www.nhc.noaa.gov/nationalsurge/ | 2026-06 | archive: `data/slosh/`, `data/slosh_nyc_full/`, `data/miami/slosh_cat*.tif` (city crops) |
| NYC Stormwater Flood Maps | NYC DEP, NYC Open Data 9i7c-xyvv | 2026 | archive: `data/pluvial/` |
| NOAA Atlas 14 vol. 10, Central Park | NOAA HDSC Precipitation Frequency Data Server | 2026 | `data/atlas14_centralpark_intensity_pds.csv` |
| MRMS MultiSensor QPE 1-h Pass 2 | NOAA, via Iowa State archive https://mtarchive.geol.iastate.edu | 2026 | `data/mrms/` (city crops) |
| ASOS one-minute precipitation, LGA, JFK, EWR | Iowa Environmental Mesonet, https://mesonet.agron.iastate.edu/request/asos/1min.phtml | 2026 | archive: `data/asos1min/` |
| Central Park hourly precipitation | NOAA ISD 725053-94728 (2005–2025, AWS); IEM ASOS (2026) | 2026 | archive: `data/asos_hourly/isd_72505394728/`; `data/asos_hourly/NYC_p01i_1948_2026.csv` |
| Airport weather reports, LGA and JFK | Iowa Environmental Mesonet | 2026 | `data/asos_hourly/wx_NYC_LGA_JFK_2020_2026.csv` |
| FloodNet flood events and sensors | NYC Open Data aq7i-eu5q (events), kb2e-tjy3 (sensors) | 2026-11-03 | `data/floodnet/` |
| 311 street-flooding complaints | NYC Open Data erm2-nwe9, descriptor "Street Flooding (SJ)" | 2026-09 | `data/sf_points.csv` |
| National Structure Inventory 2026 Base | USACE, https://nsi.sec.usace.army.mil (counties 36005, 36047, 36061, 36081, 36085; Miami-Dade 12086) | 2026 | archive: `data/nsi_nyc/nsi.parquet`, `data/miami/nsi.parquet` |
| Generic depth-damage functions | USACE EGM 01-03 (2000) and 04-01 (2003) | — | tabulated in `paper_a_core.py` |
| PLUTO | NYC DCP, NYC Open Data 64uk-42ks | 2026 | archive: `data/pluto_bsmt.csv`, `data/pluto_nyc.csv` |
| CDBG eligibility by tract (HUD LMISD 2016–2020) | NYC Open Data qmcw-ur37; HUD Exchange | 2026 | `data/nfip/tract_equity.parquet` |
| Subway stations, entrances, GTFS | MTA, data.ny.gov 39hk-dx4f and i9wp-a4ja; https://www.mta.info/developers | 2026 | `data/subway/` |
| National Flood Hazard Layer (effective) | FEMA, https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer | 2026-09 | archive: `data/fema_nfhl/nfhl_zones_nyc.geojson` |
| Preliminary FIRM, DFIRM 360497 (30 January 2015) | FEMA Flood Map Service Center | 2026-09 | archive: `data/fema_nfhl/pfirm2015_zones_nyc.geojson` |
| NFIP Redacted Policies v3, Redacted Claims v3 | OpenFEMA, https://www.fema.gov/openfema-data-page/nfip-redacted-policies-v3 and .../nfip-redacted-claims-v3 | 2026-09/10 | `data/nfip/all_current.parquet`, `data/nfip/claims_v3_sandy.json` (five boroughs); archive: `data/miami/nfip.parquet` |
| Sandy Inundation Zone | NYC Open Data 5xsi-dfpx (from the FEMA MOTF field-verified extent) | 2026 | archive: `data/sandy_zone.geojson` |
| TIGER/Line 2020 block groups, New York | US Census Bureau, tl_2020_36_bg | 2026 | archive: `data/nfip/tl_2020_36_bg.*` |

FloodNet, OpenFEMA, PLUTO and the MTA files are updated continuously; the snapshots here are those analysed. This work uses the FEMA OpenFEMA API but is not endorsed by FEMA.

## Environment

Python 3.11 with NumPy 2.2, pandas 2.3, GeoPandas 1.1, rasterio 1.4, Shapely 2.1, SciPy 1.15, statsmodels 0.14, matplotlib 3.10 and pyarrow 21 (`environment.yml`).

## Citation

Please cite the paper and the archived code (`CITATION.cff`).
