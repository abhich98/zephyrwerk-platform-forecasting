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

## Notebook 02 — Renewable Seasonality (02_renewable_seasonality)

Analysis of Germany's electricity generation mix from 2019 to 2025,
drawing on all 12 SMARD generation signals, total consumption, and
the eight neighbour market price series.

  ### Headline findings

1. **Wind and solar are strongly complementary at every timescale, but increasingly so as you aggregate.** 
   Daily wind-solar correlation is -0.41; monthly correlation rises to approximately -0.71. 
   The combined monthly series has a coefficient of variation of 18.3% — **half** of wind alone (36.6%) and
   **less than** a third of solar alone (61.4%). Adding wind to solar barely increases total monthly volatility 
   because the two sources cancel each other out seasonally. 
   This is portfolio diversification operating at the grid scale.

2. **Solar has a strong seasonal and diurnal structure.** Summer noon
   solar averages ~28 GW; winter noon averages ~9 GW — a 3× swing at
   the peak hour. Daylight window expands from ~9-15h in winter to
   ~6-20h in summer.

3. **Wind generation has a meaningful diurnal pattern (~20% variation).**
   Wind onshore averages 12.8 GW at 19-22h local but only 10.7 GW at
   10-11h, reflecting the atmospheric boundary layer effect at 100m
   hub height. Offshore wind shows the same pattern at smaller scale.

4. **Biomass and hydropower follow demand peaks.** Both show a small
   bimodal diurnal pattern (peaks at 08:00 and 19:00 local time)
   matching morning and evening consumption peaks — these are
   dispatchable renewables responding to price signals, not
   weather-driven generation.

5. **The 2021 European wind drought is visible at monthly resolution.**
   Total renewable output was meaningfully lower across most of 2021
   compared to neighbouring years (confirmed in notebook 01).

### Implications for downstream phases

- **Phase 4 ML:** Hour-of-day is a meaningful feature for both the
  price model and the renewable generation model. Wind speed should
  be used in combination with hour-of-day to capture the boundary
  layer effect. The negative wind-solar correlation suggests that
  including both as separate features (rather than a combined
  "renewable generation" feature) captures complementary information.
- **Phase 6 dashboard:** The solar month-by-hour heatmap belongs on
  the Market Monitor page. "Current renewable output vs typical for
  this hour and month" is the right framing, not "vs daily average."
- **Operational insight:** Germany's biggest renewable-light hours
  are winter mornings (10-11h, low wind + low solar). These are
  also high-demand hours, suggesting persistent import dependency
  during winter mornings is a structural feature, not weather noise.

## Notebook 03 — Consumption Patterns (03_consumption_patterns)

Seven-year analysis (2019-01-01 to 2025-12-31) of German electricity
consumption patterns, combining SMARD TOTAL_CONSUMPTION and RESIDUAL_LOAD
signals with national-average temperature derived from 4 regional
Open-Meteo series.

### Headline findings

1. **Consumption declined in two distinct shocks, not gradually.**
   The 2019-2021 baseline averaged 56-58 GW (2021 peak: 57.6 GW). The
   2022 gas crisis destroyed ~2.5 GW of demand. The 2023 nuclear
   phase-out and continued high prices took another ~2.8 GW. Since
   2024, consumption has stabilized at ~53 GW — about 3.5-4 GW below
   the pre-crisis baseline. Peak-to-trough (2021 → 2023) was 5.3 GW.

2. **The decline is concentrated in industrial activity, not residential.**
   Three independent decompositions confirm this:
   - Weekly: the weekday-Sunday gap shrunk from 12.4 GW (2019) to
     10.3 GW (2025), a 2.1 GW reduction in the "industrial signal."
   - Algebraically: weekday consumption fell 4.2 GW (60.3 → 56.1 GW);
     Sunday consumption fell only 2.1 GW (47.9 → 45.8 GW).
   - Hourly: the decline at hour 14 is 5,700 MW vs only 2,200 MW
     at hour 2 — a 2.6× difference matching the industrial activity
     profile.

3. **Winter peak vs summer trough is ~10 GW.** Jan/Feb average 60.3 GW;
   Aug averages 50.5 GW. December is *lower* than November (56.8 vs
   58.2 GW) due to Christmas/New Year industrial slowdown — the
   highest-stress periods for the grid are early January and February,
   not Christmas week.

4. **The weekly industrial signal is ~11.1 GW** (Wed peak 58.3 GW vs
   Sun trough 46.4 GW). Saturday (49.5 GW) sits between weekdays and
   Sunday, indicating substantial Saturday-shift activity in German
   manufacturing. Mon-Fri shows a clear sub-structure: Monday "warm-up"
   (~1.8 GW below Wed), Tue/Wed/Thu uniform peak operations, Friday
   "wind-down" (~1.4 GW below Wed).

5. **The diurnal peak is in the morning, not the evening.** Hour 11
   averages 62.4 GW, hour 18 only 60.3 GW. This morning-peak signature
   is characteristic of industrial-heavy load profiles, opposite to
   residential-heavy grids where the evening peak dominates. Daily
   peak-to-trough range is 18.7 GW (62.4 at 11h vs 43.7 at 3h); morning
   ramp from 05:00 to 11:00 averages 2.6 GW/hour for six consecutive
   hours.

6. **A structural "morning supply gap" exists between 05:00 and 12:00.**
   Consumption is already at 91% of its daily peak by 09:00, but solar
   production averages less than 4 GW at that hour even in summer, and
   wind is at its daily minimum (from notebook 02). This 6-hour window
   is when Germany most depends on conventional generation and imports.

7. **Consumption response to temperature is near-symmetric and V-shaped.**
   Heating elasticity below 15°C is -461 MW/°C; cooling elasticity above
   18°C is +453 MW/°C. The cooling sensitivity is essentially identical
   to heating despite Germany's historically low AC penetration —
   reflecting recent growth in residential AC adoption, data centre
   cooling, and industrial process cooling following multiple heat
   waves (2018, 2022, 2023). Total heating impact remains larger only
   because Germany's temperature range extends further below the
   comfort zone than above it.

8. **Residual load variability has grown 56% even as its mean has
   shrunk 22%.** Mean residual load fell from 37.8 GW (2019) to
   29.6 GW (2025) as renewables displaced conventional generation. Over
   the same window, standard deviation rose from 12.3 GW to 15.0 GW,
   and the coefficient of variation went from 32.5% to 50.9% — a 56%
   increase in relative volatility. This is the structural mechanism
   behind growing European price volatility: renewable variability
   propagates directly into residual load and prices.

### Implications for downstream phases

- **Phase 4 (ML):** The price model's training window restriction
  to post-2023-04-16 (established in notebook 01) is reinforced —
  residual load variability stepped up around the same time.
  Hour-of-day, day-of-week, and temperature should all be features.
  The temperature relationship contains hour/season confounding that
  interaction terms can disentangle.

- **Phase 6 (dashboard):** Three dashboard-ready charts:
  - Day-of-week × hour-of-day consumption heatmap (Historical Overview)
  - Annual mean and std of residual load (Market Monitor — the
    volatility story is the single most important operational metric)
  - Temperature V-curve (Historical Overview, optional)
  
  Recommended framing for current-conditions displays: "current value
  vs typical for this hour, day, and temperature" rather than
  "current value vs daily average."

- **Cross-cutting:** Notebook 03 connects directly to notebook 01's
  net-import finding. The morning supply gap (05:00-12:00) is when
  imports are most needed; the growing residual load volatility means
  the import need itself is highly variable. This makes neighbour-price
  spread analysis (notebook 05) the most important downstream work.