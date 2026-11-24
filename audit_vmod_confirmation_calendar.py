"""Reproduce the zero-event bounds for the frozen 33-bus P03 confirmation sample."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True, help="Directory for weeks.csv and summary.json")
args = parser.parse_args()
root = Path(__file__).resolve().parent
out = args.output
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((root / "data/vmod/profiles33_external2018/metadata.json").read_text())
start = datetime.fromisoformat(meta["window_start_utc"]).date()
protocol = root / "runs/VMOD_PROTOCOL_EXTERNAL2018_20260921"
runs = root / "runs/VMOD_EXTERNAL2018_MARGIN_0p003_20260921/confirmation/P03_pv030_svc0375"


def records(file):
    with file.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def partition(file):
    rows = records(file)
    days = {int(row["day_index"]) for row in rows}
    assert len(rows) == len(days)
    assert all(int(row["t0"]) == int(row["day_index"]) * meta["steps_per_day"] for row in rows)
    return days


confirm = partition(protocol / "confirmation_days.csv")
select = partition(protocol / "selection_days.csv")
assert len(confirm) == 349 and len(select) == 17 and not (select & confirm)
assert confirm | select == set(range(meta["days"]))
assert start + timedelta(days=meta["days"] - 1) == datetime.fromisoformat(meta["window_end_utc_inclusive"]).date()
weeks = defaultdict(list)
for day in sorted(confirm):
    date = start + timedelta(days=day)
    weeks[date - timedelta(days=date.weekday())].append(day)

events = defaultdict(dict)
for row in records(runs / "daily.csv"):
    seed, day, event = map(int, (row["seed"], row["day"], row["event"]))
    assert day not in events[seed] and event in (0, 1)
    events[seed][day] = event
assert set(events) == set(range(42, 47))
assert all(set(days) == confirm for days in events.values())
if any(sum(days.values()) for days in events.values()):
    raise ValueError("Zero-event bounds do not apply to nonzero event counts")


def zero_upper(n):
    return -math.expm1(math.log(0.05) / n)

with (out / "weeks.csv").open("w", newline="", encoding="utf-8") as handle:
    w = csv.writer(handle, lineterminator="\n")
    w.writerow(["monday_utc", "observed_days", "confirmation_dates_utc",
                *[f"event_days_seed_{seed}" for seed in sorted(events)]])
    for monday, days in weeks.items():
        w.writerow([monday, len(days), ";".join(str(start + timedelta(days=d)) for d in days),
                    *[sum(events[s][d] for d in days) for s in sorted(events)]])
counts = Counter(map(len, weeks.values()))
summary = dict(first_confirmation=str(start + timedelta(days=min(confirm))),
               last_confirmation=str(start + timedelta(days=max(confirm))),
               weeks_by_number_of_observed_days=dict(sorted(counts.items())),
               daily_upper=zero_upper(len(confirm)), observed_week_upper=zero_upper(len(weeks)),
               full_week_upper=zero_upper(counts[7]), events_by_seed={s:sum(v.values()) for s,v in events.items()})
print(json.dumps(summary, indent=2))
with (out / "summary.json").open("w", newline="\n", encoding="utf-8") as handle:
    handle.write(json.dumps(summary, indent=2) + "\n")
