# Oracle Coach — Live Tick Reporter Deployment

`EdgePredictor_LiveTickReporter.mq5` pushes the current XAUUSD M5 bar and last
bid/ask to the website every 5 seconds. The website broadcasts that to all
users connected via SSE, replacing the Yahoo `GC=F` fallback with real broker
prices.

**This EA does NOT trade.** No `OrderSend` calls. It only reads and POSTs.

---

## 1. Copy & compile

```
cp live_deployment/MQL5_Experts/EdgePredictor_LiveTickReporter.mq5 \
   "$HOME/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Experts/"
```

Open **MetaEditor** inside MT5 → open the file → F7 to compile. Should show 0
errors, 0 warnings. Produces `EdgePredictor_LiveTickReporter.ex5`.

## 2. Whitelist the POST URL

MT5 by default blocks outbound HTTP. Enable it:

**Tools → Options → Expert Advisors**
- ☑ **Allow WebRequest for listed URL**
- Add the URL (one per line):
  - `https://edgepredictor.pro` — for production
  - `http://localhost:3000` — if you want to test against `npm run dev`

Click **OK**.

## 3. Attach the EA

- Open a **XAUUSD M5** chart in MT5.
- Navigator → Expert Advisors → double-click `EdgePredictor_LiveTickReporter`.
- In the **Common** tab: ☑ Allow live trading (it doesn't trade, but needed for network).
- In the **Inputs** tab:
  - `PostUrl` — full URL, e.g. `https://edgepredictor.pro/api/tick`
  - `AuthSecret` — optional; if set, must match `TICK_SECRET` env var on the server
  - `PostEverySec` — default `5`
- OK. A smiley face icon appears top-right when running.

## 4. Verify on the server

Tail the server log. First POST arrives within ~5s:

```
POST /api/tick 200 in 42ms
```

Then visit `/lab/oracle-coach` → Live tab. The status pill should now read
**"SSE live · XAUUSD · MT5 broker"** (not "Yahoo GC=F").

## 5. Optional: set TICK_SECRET

Prevents random POSTs to `/api/tick` from polluting your live state.

On the server, add to `commercial/website/.env.local`:

```
TICK_SECRET=pick-a-random-string
```

In the EA inputs, set `AuthSecret` to the same string. Restart the EA.

Requests without the matching `X-Tick-Secret` header will 401.

---

## Troubleshooting

**EA prints "WebRequest error 4014"** → URL not whitelisted. Re-check step 2.

**EA runs but `/api/tick` never sees a POST** → firewall blocking MT5, or the
URL is reachable but returns 401/404. Check server access log.

**Live tab shows "SSE live" but source is still "Yahoo GC=F"** → MT5 tick is
older than 30 seconds (see `MT5_TICK_FRESH_MS` in `lib/market-state.ts`). The
server falls back to Yahoo when the tick goes stale (market closed, EA
crashed, etc.). If MT5 is running on an active session, check the EA journal
for errors.
