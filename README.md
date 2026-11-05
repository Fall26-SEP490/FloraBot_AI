# FloraBot candidate-only advisor

Python 3.12 FastAPI service. It has no database, payment, device or broker access.
The public API must select inventory with `flow.ai_candidates` and revalidate any
answer with `flow.ai_record_llm` before showing it. The service does not replace
that final .NET/SQL boundary.

## Configuration

- `AI_SERVICE_TOKEN`: at least 32 characters; internal Bearer credential.
- `GEMINI_API_KEY`: provider credential, passed only in the Google API request header.
- `GEMINI_MODEL`: explicit stable model ID enabled for your project. No default,
  `latest`, `preview`, experimental aliases or caller-selected model is accepted.

The application startup script generates and retains the internal token in ignored
local storage. Supply provider configuration in BE/.env. No real provider call has
been verified without credentials. `/health` reports process liveness only;
unconfigured `/suggest` returns 503 for the .NET fallback path.

## Contract

`POST /suggest` requires `Authorization: Bearer <AI_SERVICE_TOKEN>` and JSON:

```json
{
  "survey": {"recipient":"MOTHER","age":"ADULT","occasion":"BIRTHDAY","tone":"WARM","budget":300000},
  "candidates": [{"bouquet_id":"50000000-0000-0000-0000-000000000001","name":"Hoa tang me","price":250000,"tags":["MOTHER"],"description":"Hoa diu dang","brand_tone":"Am ap"}]
}
```

Response: `suggestions` (bouquet_id, reason, card_message), configured `model`,
and measured `latency_ms`. It never returns authoritative price/stock changes.
Unknown fields, duplicate candidates, over-budget candidates and invalid types
are rejected. Output IDs must be a unique subset of input. Provider errors,
blocked/truncated output, invalid output and the 1.7-second deadline return 503.
That deadline bounds the provider phase, not the full kiosk journey.

## Verification

```sh
docker build --target test -t florabot-ai-tests ./ai
```

Run from BE. Tests substitute the outbound HTTP transport, covering structured
requests, candidate membership, injection filtering, no redirects, timeout, auth,
size limits and safe errors. They do not call Gemini or prove prompt immunity.
Backend CI runs the same test image. The production image runs as UID 10001.

The .NET ai_suggest endpoint now creates the rule-based survey first, fetches up to
30 current candidates, calls this service with a 1.8-second timeout, validates the
answer and passes it to flow.ai_record_llm. Its response remains the survey UUID.
GET /api/kiosks/{kioskId}/surveys/{surveyId} returns typed results restricted to
the owning kiosk and customer (guest surveys are available to the device only).
Reads filter candidates against current inventory again. Network calls do not
hold a database transaction. Invalid/unavailable model output retains FALLBACK.
The kiosk survey UI is connected. .NET caches only SQL-approved model answers in
Valkey for 60 seconds and revalidates each reuse through SQL. Cache entries carry
no customer or session identifier. Set the same GEMINI_MODEL on API and service;
increment AI_CACHE_VERSION when changing prompt/ranking semantics. Missing model
configuration disables caching. Cache availability is optional for suggestions.
Live provider behavior and the complete end-to-end latency target remain unverified.

## Seasonal Sales Forecast (seasonal-v1)

Internal seller planning endpoint `POST /seasonal/forecast` provides deterministic demand frequency
forecasting using Poisson count regression and calendar statistics.

### Contract & Configuration

- `POST /seasonal/forecast` requires `Authorization: Bearer <AI_SERVICE_TOKEN>`.
- Request body follows frozen `seasonal-v1` contract: `history_start`, `history_end`, `target_year`,
  `target_month`, `data_source` (`DATABASE` or `SYNTHETIC`), `products` (max 100), and `observations` (max 110,000).
- Strict bounds: target month must strictly follow `history_end` (`history_end < forecast_start`), maximum
  history coverage 1096 days, maximum gap 366 days, non-negative integer counts (`invoice_count <= units`).
  Observations preceding `product.available_since` are strictly rejected with 422.
- Request body size limit: bounded up to 16MB for seasonal forecasts, 64KB for `/suggest`.
- Provider configuration: optional `GEMINI_API_KEY` and `GEMINI_MODEL`. Total client budget is 12s;
  Gemini explanation requests are bounded to at most 8s. If unconfigured, quota-limited, timed out,
  or provider returns invalid/safety-blocked output, deterministic fallback explanations are used
  (`explanation_source: "FALLBACK"`). Numerical rankings and forecasts are never altered by Gemini.

### Methodology & Modeling

- **Calendar & Proximity**: Asia/Ho_Chi_Minh dates. 4 seasons (SPRING: Mar-May, SUMMER: Jun-Aug, AUTUMN: Sep-Nov,
  WINTER: Dec-Feb). Gregorian flower holidays (Jan 1, Feb 14, Mar 8, Oct 20, Nov 20, Dec 25) with 8-day
  active windows (7 preceding days + event day).
- **Poisson Count Regression**: `scikit-learn==1.9.1` `PoissonRegressor` with L2 regularization and cyclic
  calendar features (weekday one-hot, month sin/cos, day-of-year sin/cos, holiday window flags).
  If fitting or numerical convergence fails, the model safely falls back to `WEEKDAY_BASELINE` with an explicit warning.
- **Coverage & Zero-filling**: Zero-fill is strictly limited to declared covered days
  (`max(history_start, product.available_since)` to `history_end`). Observations before `available_since` are rejected.
- **Truthful Null Forecasts**: Products with < 70 covered days (< 56 train + 14 holdout) or < 10 positive invoice
  observation days receive `INSUFFICIENT_HISTORY` status with null forecast values, retaining observed counts.
- **Holdout Validation**: Chronological split using the trailing 14 days as holdout. Out-of-sample MAE and WAPE
  are benchmarked against a trailing weekday mean baseline. The baseline is selected whenever it yields lower MAE
  or if Poisson regression encounters fitting issues.
- **Units Forecast**: Derived deterministically as `forecast_invoice_count * (observed_units / observed_invoice_count)`.
- **Annual Validation Flag**: Kept `false` in the frozen wire contract because a 14-day holdout does not validate
  multi-year annual seasonality without rolling multi-year backtests. Available annual history cycles are reported
  for reference in explanations and annotated in warnings.
- **Demand Caveat**: Warnings always include the explicit caveat that forecasts represent observed sales
  frequency rather than unconstrained true demand due to potential unobserved stockout bias.
- **Concurrency & Off-Loop Execution**: Regression training runs in worker threads (`asyncio.to_thread`) bounded
  by concurrency semaphore limited to 2 max.

### Customer Advisor Context

Existing `/suggest` customer endpoint automatically receives server-derived current Asia/Ho_Chi_Minh calendar
context (current date, season, active holiday peak) within its system prompt, preserving full backward
compatibility with all existing kiosk contracts and test suites.

### CLI Tools

Located in `tools/seasonal_cli.py`:
- `python tools/seasonal_cli.py aggregate --input sales.csv --output-json payload.json --target-year 2026 --target-month 3 --data-source DATABASE`:
  Reads raw sales transactions (order_id, product_id, date, quantity), handles UTF-8 / BOM encoding and duplicates,
  groups distinct invoice counts per day/product, and generates validated `seasonal-v1` request payloads.
  Requires explicit `--data-source` (`DATABASE` or `SYNTHETIC`) to prevent assuming provenance for arbitrary CSVs.
- `python tools/seasonal_cli.py demo --output-json demo.json --output-csv demo.csv --target-year 2026 --target-month 3`:
  Generates a deterministic 2-year synthetic dataset explicitly labeled `SYNTHETIC` demonstrating seasonal spikes, steady baselines,
  cold start products, and sparse observation archetypes.
