# Cleaned FloodNet street-flood record, New York City (Paper A)

This package holds the street-flood record used in *Recurrent flooding beyond the reach of flood protection in New York City*
(Nature Cities, submitted). It keeps every event in the source table and states, for each one, whether it passes the
checks and why not if it does not. Usable floods are in `floods_clean.csv`; removed events, each with a reason code, are in
`excluded_events.csv`; `events_flagged.csv` holds all events with every flag.

Built by `nc_package_floods.py` (Paper A code repository) on 2026-11-04.

## Sources

| Data | Identifier | Retrieved |
|---|---|---|
| FloodNet flood events (start, end, maximum depth, onset, drain and duration times) | NYC Open Data `aq7i-eu5q` | 2026-11-03 (events to 2026-10-26) |
| FloodNet sensors (location, installation, FloodNet's tidal flag) | NYC Open Data `kb2e-tjy3` | 2026-11-03 |
| Hourly radar-gauge rainfall, 1 km | NOAA MRMS MultiSensor QPE 1H Pass 2, Iowa State University archive | 2026 |
| Airport weather reports (precipitation, present weather) at Central Park, LaGuardia, JFK | Iowa Environmental Mesonet ASOS archive (METAR) | 2026-11-04 |
| Hourly water level at The Battery, relative to MHHW | NOAA CO-OPS station 8518750 (verified hourly heights) | 2026-11-04 |
| Ground elevation and open water (for distance to water) | NOAA CUDEM averaged to 55 m | 2026 |

The paper's analysis uses the events in its record snapshot (to 2026-08-11); events after that date pass the same checks
and are marked `in_paper_record = False`.

## Processing steps and counts (`step_counts.json`)

1. **Source table:** 3,581 events (FloodNet records an event for any run of one-minute depth readings above 10 mm).
2. **Paper record:** 2,929 events to 2026-08-11.
3. **Flood definition:** the water takes at least 30 min to drain from maximum depth to dry (`drain_ge_30min`). 2,127 events.
   Shorter events are mostly brief, shallow wetting during rain; on the night of 11-12 March 2026, for example, 112 events were
   recorded across the network and 9 drained in 30 min or more. The threshold is chosen on 311 complaints and the predictive
   skill of rainfall, not to remove errors; about half of the events in heavy rain also drain faster, some of them deep.
4. **Tidal flag:** FloodNet's flag plus 3 sensors that behave as tidal (`reclassified_tidal`): within 1 km of open water, at
   least 3 floods on days with less than 0.1 in of rain, at least two-thirds of them within 2 h of a Battery level above MHHW.
   The rule was fixed before results were examined. 1,739 floods at tidal sensors.
5. **Removed:** 1 flood at a sensor missing from the sensor table, and 22 floods at non-tidal sensors with no rain evidence.
6. **Counted as floods:** 2,104 (1,739 tidal, 365 rain).

Across all 3,581 events, 2,556 pass every check (`floods_clean.csv`), of which 452 come after the paper's cutoff.

## Exclusion codes (`exclusion_reason`)

| Code | Meaning | Events |
|---|---|---|
| `SHORT` | Drained from maximum depth to dry in less than 30 min; not a flood by the paper's definition | 945 |
| `NO_RAIN` | At a non-tidal sensor, with no rain at the time: radar below 0.02 in/h at the sensor and the 8 surrounding 1 km cells from 2 h before to 2 h after the start, and no precipitation reported at Central Park, LaGuardia or JFK within 1 h. Fog or mist was reported for some of these; about half started at a high tide | 29 |
| `SHORT+NO_RAIN` | Both of the above | 37 |
| `NO_SENSOR` | Sensor missing from the sensor table (location and tidal status unknown) | 14 |

Fog without rain is rare (5 of 1,116 events at non-tidal sensors) and is not a separate reason; such events fall under `NO_RAIN`.
Tidal sensors are not tested for rain, because tidal floods need none.

## Files and columns

**`events_flagged.csv`, `floods_clean.csv`, `excluded_events.csv`:** one row per event.

| Column | Definition |
|---|---|
| `sensor_id`, `sensor_name` | FloodNet identifiers |
| `flood_start_utc`, `flood_end_utc` | Event start and end as published (UTC; checked against radar timing) |
| `flood_start_new_york` | Start in New York local time |
| `max_depth_inches` | Maximum depth of the event |
| `onset_time_mins`, `drain_time_mins`, `duration_mins` | Start to maximum depth, maximum depth to dry, total |
| `duration_above_4/12/24_inches_mins` | Minutes above each depth |
| `tidal_floodnet` | FloodNet's tidal flag |
| `tidal_paper` | Tidal flag used in the paper (FloodNet's plus reclassified sensors) |
| `reclassified_tidal` | Sensor reclassified as tidal by the rule in step 4 |
| `rain_class` | `rain`, `fog, no rain`, `dry, no fog` (non-tidal sensors), or `not classed (tidal sensor)` |
| `radar_max_in_per_h_nearby` | Largest hourly radar rain at the sensor's and neighbouring cells, 2 h either side of the start |
| `airport_precip_within_1h`, `airport_fog_within_1h` | Precipitation, or fog or mist without precipitation, reported at any of the three airports within 1 h |
| `battery_max_above_mhhw_m` | Highest Battery water level above MHHW within 2 h of the start |
| `in_paper_record` | In the paper's record snapshot (to 2026-08-11) |
| `drain_ge_30min` | Meets the flood definition |
| `passes_checks` | No exclusion reason |
| `exclusion_reason` | See codes above; empty if the event passes |
| `counted_as_flood` | Used in the paper (`passes_checks` and `in_paper_record`) |

**`sensors.csv`:** one row per sensor in the sensor table, with location, installation and removal dates, FloodNet's lowest-point
height, both tidal flags, the evidence for reclassified sensors (`reclass_dry_floods`, `reclass_dry_at_high_tide`,
`reclass_to_water_m`), and counts of events in the paper record and of floods counted.

**`sensor_days_rain.csv`:** one row per rainy day (at least 0.1 in of radar rain at the sensor, New York local day) at each
non-tidal sensor from 7 days after installation, as used by the storm-and-place model.

| Column | Definition |
|---|---|
| `sensor_id`, `date_new_york` | Sensor and local day |
| `flood` | 1 if a counted flood started that day |
| `max_flood_depth_in` | Deepest counted flood that day (0 if none) |
| `max_hourly_radar_rain_in` | Maximum hourly radar rain at the sensor that day |
| `battery_above_mhhw_m_at_peak_rain`, `high_tide` | Battery level above MHHW within 2 h of the day's peak rain hour; 1 if above |
| `distance_to_water_m`, `within_1km_of_water` | Distance from the sensor to open water |

## Known limitations

- Sensors were sited where flooding is expected, so flood frequencies at sensors are upper bounds for other places.
- FloodNet does not publish when a sensor was offline; such days count as days without flooding.
- Rain evidence uses hourly radar at 1 km and three airports 5-20 km from most sensors; a brief local shower can be missed,
  so a few `NO_RAIN` events may be real rain floods.
- The network grew from 2 sensors in 2021 to about 300 in 2026; counts across years are not comparable without
  normalising by the number of sensors reporting.
- The per-minute depth profiles in the source table are not copied here.

## Terms of use and citation

FloodNet data are published on NYC Open Data under the City of New York's Open Data terms of use; please cite FloodNet and
the NYC Open Data datasets above. NOAA data (MRMS, CO-OPS, CUDEM) are public. ASOS reports are from the Iowa Environmental
Mesonet. This derived package is released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). Citation of the
paper and a DOI will be added at release.
