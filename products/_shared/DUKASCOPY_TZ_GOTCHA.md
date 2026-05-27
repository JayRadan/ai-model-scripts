# ⚠️ Dukascopy timezone gotcha — read before writing offline tests

**Short version:** when calling `dukascopy_python.fetch()` from a non-UTC
machine (e.g. Jay's Berlin laptop), always pass **timezone-aware UTC**
datetimes for `start` and `end`. Otherwise your data is shifted by the
local UTC offset and your "offline reproduction" of live results is wrong.

---

## The bug

`dukascopy_python._stream` internally does:

```python
cursor = int(start.timestamp() * 1000)
```

`.timestamp()` on a **naive** datetime treats the value as **local time**
on the calling machine. On a Berlin laptop:

- `datetime(2026, 5, 26, 12, 0, 0)` (naive) → Python interprets as 12:00 CEST
- `.timestamp()` returns the UTC epoch for "12:00 Berlin local" = **10:00 UTC**
- Dukascopy queries 10:00 UTC
- Returned ticks labeled 10:00 UTC (correct)
- You see "10:00" and conclude "Dukascopy is 2h behind" — **wrong**

The data is correct. It's your *query* that was shifted.

## The fix in every offline script

```python
from datetime import datetime, timezone, timedelta

# WRONG — naive datetime, interpreted as Berlin local time:
end = datetime.now(timezone.utc).replace(tzinfo=None)   # ← strips tz
start = end - timedelta(hours=48)

# RIGHT — tz-aware UTC throughout:
end = datetime.now(timezone.utc)
start = end - timedelta(hours=48)

df = dukascopy_python.fetch(
    instrument="XAU/USD",
    interval=dukascopy_python.INTERVAL_TICK,
    offer_side=dukascopy_python.OFFER_SIDE_BID,
    start=start, end=end,   # both tz-aware UTC
)

# Returned df.index is tz-aware UTC. For comparison with naive
# server bar timestamps, strip the tz AFTER fetch:
df = df.reset_index().rename(columns={"timestamp": "time"})
df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
```

## Why the live server is unaffected

Render's containers run with `TZ=UTC` (default). So `datetime.now()` and
`.replace(tzinfo=None).timestamp()` produce the correct UTC epoch on the
server. The bug only manifests on non-UTC dev machines.

You can confirm with:
```bash
timedatectl | grep "Time zone"
# Render container: Etc/UTC
# Jay's laptop:     Europe/Berlin (CEST, +0200)
```

## When you wrote this script — the symptom

On 2026-05-27 we spent ~2 hours hypothesizing fake production bugs:

- "Dukascopy has a 2h tick lag" — false
- "Server orderflow features are all zero" — false
- "Need to subscribe to Polygon.io for live ticks" — false
- "Need EA to stream ticks to the server" — false

All ruled out once we passed `tzinfo=timezone.utc` to a test fetch and saw
the returned timestamps shift by exactly 120 minutes. The live server was
always working correctly.

**Trigger to re-check this gotcha:**

- You wrote a script in `/tmp/` that fetches from Dukascopy
- Offline Q values don't match live Q values from the funnel
- The mismatch is consistent (constant offset, not noise)
- The offset is suspiciously close to your local TZ offset (1h CET / 2h CEST)

If all four → you have the TZ bug, not a server bug.

## Checklist for any new offline diagnostic

- [ ] Use `datetime.now(timezone.utc)` — tz-aware, not naive
- [ ] Pass tz-aware datetimes to `dukascopy_python.fetch()`
- [ ] Strip tz from returned `df.index` only when needed for join with naive bar times
- [ ] Sanity check: print `df["time"].min()` and verify it matches what you asked for
- [ ] If comparing against live funnel `bar_ts`, the funnel times are **UTC** (server is UTC) — so your offline times must also be UTC

## See also

- Memory: `dukascopy_tz_pitfall.md`
- `commercial/server/decision_engine/tick_source.py` — production code (unaffected, server is UTC)
- `commercial/server/decision_engine/dukascopy_source.py` — production bar fetcher (unaffected)
