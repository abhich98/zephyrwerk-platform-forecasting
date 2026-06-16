# Zephyrwerk Energy Analytics — Phase 2 Findings

This is the master findings document for Phase 2 EDA. Each section
corresponds to one notebook in `notebooks/eda/`. For per-finding
implementation details, see the individual notebooks. For data
quality rules feeding Phase 3 dbt tests, see `DATA_QUALITY.md`.

**Methodology note:** Generation totals reported here include pumped
storage discharge (consistent with conventional reporting by AGEE-Stat
and Eurostat). Renewable share calculations exclude pumped storage
from both numerator and denominator — treating it as storage, not
primary generation — and use energy-weighted methodology rather than
the mean of hourly shares.

## Data quality baseline (00_data_audit)

Seven-year dataset (2019-01-01 to 2025-12-31), 2,557 days, ~2.6M rows.

- **Coverage:** 22 of 23 SMARD signals and all weather signals are
  complete to the row. Only Nuclear (signal retired 4 February 2024)
  and Czechia (single missing day, Christmas 2019) deviate.
- **Quality:** Zero duplicates, zero schema violations, all units
  consistent per signal. All negative values and zeros are physically
  meaningful — no measurement errors detected.
- **Two structural data events documented in `DATA_QUALITY.md`:**
  - Nuclear phase-out and signal retirement (April 2023 / February 2024)
  - Czechia price missing 2019-12-25 (likely OTE publishing gap)

## Notebook 01 — Energy mix evolution (01_energy_mix_history)

Analysis of Germany's electricity generation mix from 2019 to 2025,
drawing on all 12 SMARD generation signals, total consumption, and
the eight neighbour market price series.

### Headline findings

1. **Germany became a net electricity importer in 2023.** From 2019
   through 2022, Germany was a consistent net exporter (annual average
   surplus ~2 GW). In 2023 the balance flipped, coinciding with the
   completion of the nuclear phase-out in April. Germany has since
   stabilized at ~3 GW continuous net import. The mechanism: domestic
   generation declined ~9.7 GW (59.6 → 49.9 GW) while consumption fell
   only ~3.8 GW (56.8 → 53.0 GW). The 5.9 GW gap is filled by imports.

2. **Nuclear: complete and rapid exit.** From ~8 GW average baseload
   in 2019 to zero in 9 months. The final three reactors shut down on
   15 April 2023; SMARD continued publishing value=0 until 30 January
   2024, then nulls for 132 hours before the signal was retired
   entirely on 4 February 2024 at 22:00 UTC.

3. **Renewable share grew from 43.6% to 60.2%, but ~80% of the 2023
   jump came from denominator effects.** The 9.1-point increase from
   2022 (47.6%) to 2023 (56.7%) decomposes as: ~7.3 points from nuclear
   leaving the denominator plus fossil decline, only ~1.8 points from
   actual renewable growth (renewables added ~2 GW that year). The
   headline "60% renewable" figure obscures that recent gains come
   more from baseload retirement than from capacity expansion.

4. **Renewable absolute generation has plateaued.** Total renewable
   generation grew from ~25 GW (2019) to ~29 GW (2025), but added
   only ~270 MW between 2024 and 2025. Wind onshore has declined two
   years running (13.6 GW in 2023 → 12.8 in 2024 → 12.1 in 2025);
   only solar growth (4.8 → 8.4 GW over the window, +76%) has
   prevented total renewable decline.

5. **Coal decline was delayed by the 2022 crisis.** From 2019 to 2022,
   total coal stayed at 14-18 GW average — the gas crisis kept fossil
   generation high because coal was suddenly cheaper than gas.
   Structural decline only began in 2023, dropping total coal
   generation from ~15 GW (2022) to ~8 GW (2025).

6. **Natural gas is the only fossil source that grew.** From 6.2 GW
   (2019) to 6.9 GW (2025). Despite the 2022 price crisis temporarily
   making gas unaffordable, its structural role as a flexible
   balancing technology to variable renewables has kept it expanding
   while coal declined sharply. This reflects a system-level shift:
   as renewables grow, the grid needs more ramping flexibility,
   not less.

7. **Total generation declined by ~9.7 GW, almost exactly matching
   the lost nuclear capacity.** This is the strongest evidence that
   imports — not new domestic capacity — filled the nuclear gap.
   Renewable additions (+4 GW combined wind + solar) only partially
   offset nuclear's exit (-8 GW), with fossil decline (-6 GW from
   peak) compounding the gap.

8. **The 2021 European wind drought is the only year-over-year
   renewable decline in the window.** Renewable generation fell
   ~2 GW from 2020 to 2021, driven by anomalously low wind speeds
   across northwest Europe through most of 2021. Recovery in 2022
   confirmed it was weather, not capacity.

### Implications for downstream phases

- **Phase 3 (dbt):** All structural breaks identified in this
  notebook are captured in `DATA_QUALITY.md` as staging-model rules.
  No further audit work required before building the transformation
  layer.

- **Phase 4 (ML):** The price forecasting model's training window
  should start no earlier than 2023-04-16 (post-nuclear regime).
  Pre-phase-out data represents a structurally different grid that
  no longer exists in serving time. Additionally, neighbour-price
  spreads are likely strong features given Germany's structural
  import dependency — this should be tested explicitly during
  feature engineering.

- **Phase 6 (dashboard):** Four dashboard-ready visualizations
  exist as notebook outputs:
  - Monthly stacked-area energy mix (Historical Overview page)
  - Nuclear phase-out daily detail with annotated structural breaks
  - Coal generation monthly stacked with crisis-line annotation
  - 2019 vs 2025 side-by-side mix comparison

- **Business framing:** The original PLATFORM.md framing of
  "Zephyrwerk needs price forecasts to time market sales" is
  sharpened by these findings into "Zephyrwerk operates in a market
  that is structurally short of domestic generation, where prices
  are increasingly coupled to import availability from neighbours"
  — making neighbour-spread analysis (notebook 05) the most
  strategically important downstream work.