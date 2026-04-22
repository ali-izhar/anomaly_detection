# MIT Reality Mining — raw data

`Proximity.csv` is the Bluetooth-proximity trace from the MIT Reality Mining project (Eagle & Pentland, 2004), sliced to the 289-day academic-year window used in Ali & Ho (ICDM 2025) §V.

## Schema

| Column | Type | Meaning |
|---|---|---|
| `user.id` | int | Anchor phone identifier |
| `remote.user.id.if.known` | int or NA | Observed peer (NA if out-of-study phone) |
| `time` | datetime | Observation timestamp |
| `prob2` | float in [0,1] or NA | Proximity probability (RSSI-derived) |

## Loading

```python
from hmd.data.mit_reality import load, MIT_EVENTS

seq, meta = load()   # picks up Proximity.csv from this directory
# seq.graphs       → list of 289 daily nx.Graph snapshots, day-0 = 2008-09-19
# seq.change_points → [23, 68, 97, 173, 234] — paper Table II events
# MIT_EVENTS       → {23: "Columbus Day / Fall Break", ..., 234: "End of Spring Semester"}
```

See `hmd/data/mit_reality.py` for aggregation semantics (24-h edges, threshold, day alignment).

## Source

Download the raw CSV from [MIT Reality Commons](http://realitycommons.media.mit.edu/realitymining.html) if the file is missing.
