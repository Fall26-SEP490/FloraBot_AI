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
