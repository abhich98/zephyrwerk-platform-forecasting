import csv

import holidays

de_holidays = holidays.Germany(years=range(2019, 2031), subdiv=None)

with open("dbt/seeds/german_public_holidays.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "holiday_name"])
    for date, name in sorted(de_holidays.items()):
        writer.writerow([date, name])