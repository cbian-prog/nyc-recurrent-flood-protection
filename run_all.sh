#!/usr/bin/env bash
# Reproduce every analysis, figure and table of the paper, in dependency order.
# Run from the repository root after unpacking the Zenodo input archive into it (see README.md).
set -euo pipefail
run() { echo "== $*"; python "$@"; }

# Optional downloads (their outputs are already included or in the input archive)
# run download_pfirm.py             # 2015 preliminary FIRM from FEMA
# run nfip_sandy_claims_v3.py       # Hurricane Sandy NFIP claims from OpenFEMA
# run naccs_jpm_points.py           # needs the USACE CHS per-save-point files (registration); output included

# 1 Surge model
run naccs_aai.py
run paper_a_core.py
run paper_a_mc.py
run figED1_validation.py
run figED5_vulnerability.py
run nc_valuation_sensitivity.py

# 2 Areas
run nc_areas.py

# 3 Street-flood record
run nc_fog_check.py
run nc_tidal_reclass.py
run nc_data.py
run nc_si_transfer.py
run nc_si_subhourly.py
run nc_si_duration.py
run nc_storm_stats.py
run nc_rain_climatology.py
run nc_storm_model.py
run nc_tide_effect.py
run nc_package_floods.py

# 4 Exposure
run basement_exposure.py
run figED6_basements.py
run subway_flood.py
run slr_overlay.py
run fig2_impact_rp.py

# 5 Insurance
run fig5_gap_nsi.py
run figED7_nfip.py
run pluvial_insurance_nsi.py
CITY=nyc run insurance_city.py
CITY=miami run insurance_city.py
run figED3_hazard_basis.py
run nc_si_sensitivity.py

# 6 Figures and tables
for s in nc_fig1.py nc_fig2.py nc_fig3.py nc_fig4_ed78.py nc_edfig_record.py nc_edfig_surge.py nc_ed456.py \
         nc_si_robust.py nc_si_placement.py nc_supp.py make_supp_tables.py; do run "$s"; done
