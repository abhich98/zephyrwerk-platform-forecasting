## Finding 01 — Nuclear generation: structural break + signal retirement (updated)

**Scope:** SMARD signal `NUCLEAR` (filter 1224), entire dataset.

**Four-regime lifecycle:**
1. 2019-01-01 → 2023-04-14: normal generation, ~2,600 MW daily average
2. 2023-04-15: phase-out day, partial generation (~2,129 MW)
3. 2023-04-16 → 2024-01-30 10:00 UTC: zero values (reactors offline, signal still published)
4. 2024-01-30 11:00 → 2024-02-04 22:00 UTC: nulls (132 hours, wind-down)
5. 2024-02-04 23:00 UTC onward: signal retired, no rows
- Total rows: 44,663 vs. expected 61,368

**Expected:** Real-world data, not pipeline behavior. Germany's last three
reactors shut down 15 April 2023. SMARD wound down the series with ~5.5
days of nulls before retiring it.

**Phase 3 (dbt) action:** In `stg_smard_generation`:
  - `coalesce(nuclear_value, 0)` to treat the 132 wind-down nulls as zeros
    (they are physically zero — reactors are offline)
  - Restrict the model's valid window to `timestamp <= '2024-02-04 22:00 UTC'`
  - No `not_null` test on NUCLEAR in the analytics layer
  - Use left join in `fct_energy_generation` to accept post-cutoff absence

**Phase 4 (ML) action:** Strong candidate to drop NUCLEAR from price-model
features entirely. From April 16 2023 onward (the regime the model needs to
predict) the value is structurally 0 or missing — it adds no information.
Final decision in Phase 4.

**Dashboard (Phase 6) note:** Render the phase-out deliberately on the
Historical Overview page. The four-regime transition is one of the
strongest narrative beats in the dataset.

## Finding 02 — Czechia price: single missing day

**Scope:** SMARD signal `CZECHIA` (filter 261), 2019-12-25 only.

**Observation:** All 24 hours of 2019-12-25 are fully absent. Every other day
in 2019-01-01 → 2025-12-31 has the expected 24 hours. Other neighbour price
signals (Austria, France, Netherlands, Poland, Switzerland, Denmark 1, Denmark 2)
have complete data for that date — the gap is Czechia-specific.

**Expected:** Publication gap on the Czech side (likely OTE not publishing on
Christmas Day 2019). Single-day outage, never backfilled by SMARD. Not a
pipeline bug.

**Phase 3 (dbt) action:** Allow CZECHIA to have one missing day; do not enforce
strict 100% daily completeness. Either accept nulls in the price-spread fact
table, or forward-fill from 2019-12-24 with a `_imputed` flag column.

**Phase 4 (ML) action:** Negligible (1 day / 2557 ≈ 0.04% of training data).
No special handling needed.

## Note — Solar generation: twilight values are real

`SOLAR` (filter 4068) shows exact zeros only for ~13% of hours; another
~25% of hours have small-but-nonzero values (0 < value < 100 MW).
This is physically correct — solar PV captures dawn/dusk twilight and
small distributed installations respond to ambient light below sunset
thresholds.

**Downstream guidance:** if a model or chart needs a "daylight active"
filter, use a threshold (e.g. value > 100 MW), not value > 0.

## Note — Wind direction is circular

`wind_direction_100m` (all 4 regions) is a circular variable: 0° and 360°
are the same direction (north). Models using this feature must encode
it as sin/cos pair rather than raw degrees, or distance metrics will
treat 359° and 1° as opposite directions when they're nearly identical.