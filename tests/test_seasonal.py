import calendar
import json
from datetime import date, timedelta
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from app import Settings, create_app
from seasonal import (
    DEMAND_CAVEAT,
    MONTH_TO_SEASON,
    DailyObservation,
    ProductInput,
    SeasonalForecastRequest,
    calculate_forecast_statistics,
    extract_date_features,
    get_current_vietnam_calendar_context,
    get_target_month_holidays,
    is_holiday_event_day,
    is_in_holiday_window,
)

TOKEN = "test-service-token-with-at-least-32-bytes"
PROD_1 = "60000000-0000-0000-0000-000000000001"
PROD_2 = "60000000-0000-0000-0000-000000000002"
PROD_COLD = "60000000-0000-0000-0000-000000000003"
SETTINGS = Settings(TOKEN, "test-provider-key", "gemini-2.5-flash-lite")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def sample_request_payload(
    history_days=90,
    target_year=2026,
    target_month=3,
    prod_id=PROD_1,
    available_days_ago=None,
    pos_obs_days=25,
):
    target_start = date(target_year, target_month, 1)
    history_end = target_start - timedelta(days=1)
    history_start = history_end - timedelta(days=history_days - 1)

    available_since = (
        history_end - timedelta(days=available_days_ago - 1)
        if available_days_ago is not None
        else history_start
    )

    products = [
        {
            "product_id": prod_id,
            "name": "Hoa Hồng Đà Lạt",
            "available_since": available_since.isoformat(),
        }
    ]

    observations = []
    # Generate observations spread across history
    step = max(1, history_days // (pos_obs_days + 1))
    current = history_start
    obs_count = 0
    while current <= history_end and obs_count < pos_obs_days:
        if current >= available_since:
            observations.append(
                {
                    "day": current.isoformat(),
                    "product_id": prod_id,
                    "invoice_count": 2,
                    "units": 4,
                }
            )
            obs_count += 1
        current += timedelta(days=step)

    return {
        "contract_version": "seasonal-v1",
        "history_start": history_start.isoformat(),
        "history_end": history_end.isoformat(),
        "target_year": target_year,
        "target_month": target_month,
        "data_source": "SYNTHETIC",
        "products": products,
        "observations": observations,
        "include_explanations": False,
    }


def gemini_explanation_response(items):
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": json.dumps({"explanations": items})}]
                    },
                }
            ]
        },
    )


def test_calendar_seasons_mapping():
    assert MONTH_TO_SEASON[3] == "SPRING"
    assert MONTH_TO_SEASON[4] == "SPRING"
    assert MONTH_TO_SEASON[5] == "SPRING"
    assert MONTH_TO_SEASON[6] == "SUMMER"
    assert MONTH_TO_SEASON[7] == "SUMMER"
    assert MONTH_TO_SEASON[8] == "SUMMER"
    assert MONTH_TO_SEASON[9] == "AUTUMN"
    assert MONTH_TO_SEASON[10] == "AUTUMN"
    assert MONTH_TO_SEASON[11] == "AUTUMN"
    assert MONTH_TO_SEASON[12] == "WINTER"
    assert MONTH_TO_SEASON[1] == "WINTER"
    assert MONTH_TO_SEASON[2] == "WINTER"


def test_calendar_leap_years():
    # 2024 is leap year (29 days in Feb), 2025 is not (28 days)
    assert calendar.monthrange(2024, 2)[1] == 29
    assert calendar.monthrange(2025, 2)[1] == 28

    holidays_2024 = get_target_month_holidays(2024, 2)
    assert any(h.code == "VALENTINE" for h in holidays_2024)
    assert holidays_2024[0].day == date(2024, 2, 14)


def test_gregorian_holiday_windows():
    # Valentine Feb 14: window is Feb 7 to Feb 14 (8 days)
    assert not is_in_holiday_window(date(2026, 2, 6))
    assert is_in_holiday_window(date(2026, 2, 7))
    assert is_in_holiday_window(date(2026, 2, 14))
    assert not is_in_holiday_window(date(2026, 2, 15))
    assert is_holiday_event_day(date(2026, 2, 14))
    assert not is_holiday_event_day(date(2026, 2, 13))

    # Jan 1 window: Dec 25 to Jan 1
    assert is_in_holiday_window(date(2025, 12, 25))
    assert is_in_holiday_window(date(2026, 1, 1))
    assert not is_in_holiday_window(date(2026, 1, 2))


def test_vietnam_calendar_context_format():
    context = get_current_vietnam_calendar_context()
    assert "Thời gian hiện tại tại Việt Nam" in context
    assert any(
        season in context for season in ("Mùa Xuân", "Mùa Hạ", "Mùa Thu", "Mùa Đông")
    )


def test_zero_filling_only_declared_covered_days():
    # Product available_since = history_start + 10 days
    # Days before available_since must NOT be zero-filled
    h_start = date(2025, 1, 1)
    h_end = date(2025, 3, 31)
    avail = date(2025, 1, 11)

    req = SeasonalForecastRequest(
        contract_version="seasonal-v1",
        history_start=h_start,
        history_end=h_end,
        target_year=2025,
        target_month=4,
        data_source="DATABASE",
        products=[
            ProductInput(product_id=UUID(PROD_1), name="Test", available_since=avail)
        ],
        observations=[
            DailyObservation(
                day=avail + timedelta(days=i * 5),
                product_id=UUID(PROD_1),
                invoice_count=1,
                units=1,
            )
            for i in range(12)
        ],
        include_explanations=False,
    )

    resp = calculate_forecast_statistics(req)
    prod = resp.products[0]
    expected_covered = (h_end - avail).days + 1
    assert (
        prod.validation.training_days + prod.validation.holdout_days == expected_covered
    )


def test_insufficient_history_short_coverage():
    # Product coverage < 70 days -> INSUFFICIENT_HISTORY, forecast is None
    payload = sample_request_payload(
        history_days=50, available_days_ago=50, pos_obs_days=15
    )
    req = SeasonalForecastRequest.model_validate(payload)
    resp = calculate_forecast_statistics(req)

    prod = resp.products[0]
    assert prod.model_status == "INSUFFICIENT_HISTORY"
    assert prod.forecast_invoice_count is None
    assert prod.forecast_units is None
    assert prod.validation.mae is None
    assert prod.observed_invoice_count == 15 * 2
    assert prod.observed_units == 15 * 4
    assert not prod.annual_seasonality_validated
    assert any("insufficient history" in w.lower() for w in resp.warnings)


def test_insufficient_history_few_observations():
    # Product coverage 90 days but only 5 positive invoice days (< 10)
    payload = sample_request_payload(history_days=90, pos_obs_days=5)
    req = SeasonalForecastRequest.model_validate(payload)
    resp = calculate_forecast_statistics(req)

    prod = resp.products[0]
    assert prod.model_status == "INSUFFICIENT_HISTORY"
    assert prod.forecast_invoice_count is None
    assert prod.forecast_units is None


def test_chronological_split_and_no_data_leakage():
    # Verify holdout is strictly the last 14 days and training is history_end - 14
    history_days = 100
    payload = sample_request_payload(history_days=history_days, pos_obs_days=30)
    req = SeasonalForecastRequest.model_validate(payload)
    resp = calculate_forecast_statistics(req)

    prod = resp.products[0]
    assert prod.validation.holdout_days == 14
    assert prod.validation.training_days == history_days - 14
    assert prod.model_status in ("POISSON_REGRESSION", "WEEKDAY_BASELINE")
    assert prod.forecast_invoice_count is not None
    assert prod.forecast_invoice_count >= 0.0
    assert prod.forecast_units is not None
    assert prod.forecast_units >= prod.forecast_invoice_count


def test_deterministic_forecast_nonnegative_and_demand_caveat():
    payload = sample_request_payload(history_days=120, pos_obs_days=40)
    req = SeasonalForecastRequest.model_validate(payload)
    resp = calculate_forecast_statistics(req)

    assert resp.contract_version == "seasonal-v1"
    assert resp.aggregation_level == "PRODUCT"
    assert resp.warnings[0] == DEMAND_CAVEAT
    prod = resp.products[0]
    assert prod.forecast_invoice_count >= 0.0
    assert prod.forecast_units >= prod.forecast_invoice_count
    # Units ratio holds
    ratio = prod.observed_units / prod.observed_invoice_count
    assert (
        abs(prod.forecast_units - round(prod.forecast_invoice_count * ratio, 2)) < 0.05
    )


def test_annual_seasonality_validation_flag():
    # In accordance with specification: 14-day holdout does NOT validate annual seasonality.
    # annual_seasonality_validated must be False; history cycles and warnings are reported.
    payload_long = sample_request_payload(history_days=730, pos_obs_days=50)
    req_long = SeasonalForecastRequest.model_validate(payload_long)
    resp_long = calculate_forecast_statistics(req_long)
    assert resp_long.products[0].annual_seasonality_validated is False
    assert any("Annual seasonality is not validated" in w for w in resp_long.warnings)

    payload_short = sample_request_payload(history_days=100, pos_obs_days=30)
    req_short = SeasonalForecastRequest.model_validate(payload_short)
    resp_short = calculate_forecast_statistics(req_short)
    assert resp_short.products[0].annual_seasonality_validated is False
    assert any("Annual seasonality is not validated" in w for w in resp_short.warnings)


def test_feature_extraction_length_and_determinism():
    d = date(2026, 3, 8)
    feats1 = extract_date_features(d)
    feats2 = extract_date_features(d)
    assert feats1 == feats2
    assert len(feats1) == 13
    # Sunday is index 6
    assert feats1[d.weekday()] == 1.0
    # Mar 8 is Women's Day
    assert feats1[-1] == 1.0  # is_holiday_event
    assert feats1[-2] == 1.0  # is_holiday_window


def test_api_auth_and_health():
    app = create_app(SETTINGS)
    with TestClient(app) as client:
        # Health check open
        assert client.get("/health").json() == {"status": "ok"}

        # Unauthorized
        payload = sample_request_payload()
        assert client.post("/seasonal/forecast", json=payload).status_code == 401
        assert (
            client.post(
                "/seasonal/forecast",
                headers={"Authorization": "Bearer wrong-token"},
                json=payload,
            ).status_code
            == 401
        )

        # Authorized
        res = client.post("/seasonal/forecast", headers=HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["contract_version"] == "seasonal-v1"
        assert len(data["products"]) == 1


def test_api_unconfigured_service_token():
    bad_settings = Settings("short-token", SETTINGS.api_key, SETTINGS.model)
    app = create_app(bad_settings)
    with TestClient(app) as client:
        res = client.post(
            "/seasonal/forecast",
            headers={"Authorization": "Bearer short-token"},
            json=sample_request_payload(),
        )
        assert res.status_code == 503


@pytest.mark.parametrize(
    "corrupt",
    [
        "empty_products",
        "duplicate_products",
        "unknown_product_in_obs",
        "obs_before_available_since",
        "history_reversed",
        "history_too_long",
        "target_in_past",
        "gap_too_large",
        "duplicate_obs",
        "invoice_count_gt_units",
        "negative_count",
    ],
)
def test_contract_validation_rejects_corrupt_payload(corrupt):
    payload = sample_request_payload()
    if corrupt == "empty_products":
        payload["products"] = []
    elif corrupt == "duplicate_products":
        payload["products"].append(payload["products"][0])
    elif corrupt == "unknown_product_in_obs":
        payload["observations"].append(
            {
                "day": payload["history_end"],
                "product_id": PROD_2,
                "invoice_count": 1,
                "units": 1,
            }
        )
    elif corrupt == "obs_before_available_since":
        # Make product available only on history_end, while observations exist before it
        payload["products"][0]["available_since"] = payload["history_end"]
    elif corrupt == "history_reversed":
        payload["history_start"] = payload["history_end"]
        payload["history_end"] = (
            date.fromisoformat(payload["history_start"]) - timedelta(days=10)
        ).isoformat()
    elif corrupt == "history_too_long":
        payload["history_start"] = (
            date.fromisoformat(payload["history_end"]) - timedelta(days=1100)
        ).isoformat()
    elif corrupt == "target_in_past":
        # target_month 1 on 2026 when history ends 2026-02-28
        payload["target_year"] = 2026
        payload["target_month"] = 2
    elif corrupt == "gap_too_large":
        payload["target_year"] = 2028
        payload["target_month"] = 1
    elif corrupt == "duplicate_obs":
        payload["observations"].append(payload["observations"][0])
    elif corrupt == "invoice_count_gt_units":
        payload["observations"][0]["invoice_count"] = 10
        payload["observations"][0]["units"] = 5
    elif corrupt == "negative_count":
        payload["observations"][0]["invoice_count"] = -1

    app = create_app(SETTINGS)
    with TestClient(app) as client:
        res = client.post("/seasonal/forecast", headers=HEADERS, json=payload)
        assert res.status_code == 422


def test_gemini_enrichment_success():
    def provider(request):
        assert request.headers["x-goog-api-key"] == SETTINGS.api_key
        return gemini_explanation_response(
            [
                {
                    "product_id": PROD_1,
                    "explanation": "Cân nhắc dịp 8-3 cùng số liệu bán hàng được cung cấp.",
                }
            ]
        )

    payload = sample_request_payload(pos_obs_days=25)
    payload["include_explanations"] = True

    app = create_app(SETTINGS, transport=httpx.MockTransport(provider))
    with TestClient(app) as client:
        res = client.post("/seasonal/forecast", headers=HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()
        prod = data["products"][0]
        assert prod["explanation_source"] == "GEMINI"
        assert "Cân nhắc dịp 8-3" in prod["explanation"]


def test_gemini_unknown_id_rejected_and_falls_back():
    def provider(request):
        # Gemini returns an unknown foreign product_id
        return gemini_explanation_response(
            [
                {
                    "product_id": "00000000-0000-0000-0000-000000000099",
                    "explanation": "Bịa đặt sản phẩm không có trong danh mục.",
                }
            ]
        )

    payload = sample_request_payload(pos_obs_days=25)
    payload["include_explanations"] = True

    app = create_app(SETTINGS, transport=httpx.MockTransport(provider))
    with TestClient(app) as client:
        res = client.post("/seasonal/forecast", headers=HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()
        prod = data["products"][0]
        # Unknown ID ignored, kept fallback explanation
        assert prod["explanation_source"] == "FALLBACK"
        assert prod["forecast_invoice_count"] is not None


@pytest.mark.parametrize(
    "unsupported_text",
    [
        "sụt giảm mạnh so với cùng kỳ năm trước",
        "Dự báo tăng nhẹ trong tháng 11",
        "Nhu cầu theo chu kỳ hàng năm",
        "Tăng trưởng 25% so với năm ngoái",
        "Higher sales compared to prior year (+10% YoY)",
        "Dự báo suy giảm 15% so với mùa trước",
    ],
)
def test_gemini_unsupported_comparison_rejected_and_falls_back(unsupported_text):
    def provider(request):
        return gemini_explanation_response(
            [
                {
                    "product_id": PROD_1,
                    "explanation": unsupported_text,
                }
            ]
        )

    payload = sample_request_payload(pos_obs_days=25)
    payload["include_explanations"] = True

    app = create_app(SETTINGS, transport=httpx.MockTransport(provider))
    with TestClient(app) as client:
        res = client.post("/seasonal/forecast", headers=HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()
        prod = data["products"][0]
        # Unsupported YoY/percentage comparison must be rejected and keep deterministic fallback
        assert prod["explanation_source"] == "FALLBACK"
        assert prod["forecast_invoice_count"] is not None
        assert unsupported_text not in prod["explanation"]


@pytest.mark.parametrize(
    "mock_resp",
    [
        httpx.Response(429, text="Rate limit exceeded"),
        httpx.Response(500, text="Internal error"),
        httpx.Response(200, text="malformed json"),
        httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}),
        httpx.Response(200, json={"candidates": []}),
    ],
)
def test_gemini_failures_safely_fallback_without_altering_numbers(mock_resp):
    payload = sample_request_payload(pos_obs_days=25)
    payload["include_explanations"] = True

    app = create_app(SETTINGS, transport=httpx.MockTransport(lambda _: mock_resp))
    with TestClient(app) as client:
        res = client.post("/seasonal/forecast", headers=HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()
        prod = data["products"][0]
        assert prod["explanation_source"] == "FALLBACK"
        assert prod["forecast_invoice_count"] is not None
        assert prod["forecast_units"] is not None


def test_body_limit_enforced_on_both_endpoints():
    app = create_app(SETTINGS)
    with TestClient(app) as client:
        # /suggest limited to 64KB with valid token
        res_suggest = client.post("/suggest", headers=HEADERS, content=b"x" * 65537)
        assert res_suggest.status_code == 413

        # /suggest unauthenticated with malformed body returns 401 before body parsing
        res_suggest_unauth = client.post("/suggest", content=b"not json")
        assert res_suggest_unauth.status_code == 401

        # /suggest authenticated with malformed body returns 422
        res_suggest_bad = client.post("/suggest", headers=HEADERS, content=b"not json")
        assert res_suggest_bad.status_code == 422

        # /seasonal/forecast allows > 64KB (e.g. 100KB), but unauthenticated returns 401 before body parsing
        res_seasonal_100k = client.post(
            "/seasonal/forecast",
            headers={"Authorization": "Bearer wrong"},
            content=b"x" * 100000,
        )
        assert res_seasonal_100k.status_code == 401

        # Authenticated with 100KB invalid body returns 422
        res_seasonal_bad = client.post(
            "/seasonal/forecast",
            headers=HEADERS,
            content=b"x" * 100000,
        )
        assert res_seasonal_bad.status_code == 422

        # Excessively huge payload (> 16MB) rejected with 413
        res_seasonal_huge = client.post(
            "/seasonal/forecast",
            headers=HEADERS,
            content=b"x" * (16 * 1024 * 1024 + 1),
        )
        assert res_seasonal_huge.status_code == 413


def test_forecast_concurrency_bounded_to_2():
    from app import FORECAST_CONCURRENCY

    assert FORECAST_CONCURRENCY._value == 2


def test_poisson_forced_fit_failure_falls_back_to_weekday_baseline(monkeypatch):
    import seasonal
    from sklearn.linear_model import PoissonRegressor

    def fail_fit(self, *args, **kwargs):
        raise ValueError("Simulated Poisson numerical/convergence failure")

    monkeypatch.setattr(PoissonRegressor, "fit", fail_fit)

    payload = sample_request_payload(history_days=100, pos_obs_days=30)
    req = SeasonalForecastRequest.model_validate(payload)
    resp = seasonal.calculate_forecast_statistics(req)

    prod = resp.products[0]
    assert prod.model_status == "WEEKDAY_BASELINE"
    assert prod.forecast_invoice_count is not None
    assert prod.forecast_invoice_count >= 0.0
    assert any("fell back to weekday baseline" in w for w in resp.warnings)
    # Ensure no raw exception strings leak into user warnings
    assert not any("Simulated Poisson" in w for w in resp.warnings)


def test_poisson_convergence_warning_falls_back_to_weekday_baseline(monkeypatch):
    import warnings
    import seasonal
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import PoissonRegressor

    orig_fit = PoissonRegressor.fit

    def warn_fit(self, X, y, sample_weight=None):
        warnings.warn("Simulated ConvergenceWarning", ConvergenceWarning)
        return orig_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(PoissonRegressor, "fit", warn_fit)

    payload = sample_request_payload(history_days=100, pos_obs_days=30)
    req = SeasonalForecastRequest.model_validate(payload)
    resp = seasonal.calculate_forecast_statistics(req)

    prod = resp.products[0]
    assert prod.model_status == "WEEKDAY_BASELINE"
    assert prod.forecast_invoice_count is not None
    assert any("fell back to weekday baseline" in w for w in resp.warnings)
    assert not any("Simulated ConvergenceWarning" in w for w in resp.warnings)


def test_poisson_nonfinite_holdout_prediction_falls_back_to_weekday_baseline(
    monkeypatch,
):
    import seasonal
    from sklearn.linear_model import PoissonRegressor

    def nonfinite_predict(self, X):
        return [float("nan")] * len(X)

    monkeypatch.setattr(PoissonRegressor, "predict", nonfinite_predict)

    payload = sample_request_payload(history_days=100, pos_obs_days=30)
    req = SeasonalForecastRequest.model_validate(payload)
    resp = seasonal.calculate_forecast_statistics(req)

    prod = resp.products[0]
    assert prod.model_status == "WEEKDAY_BASELINE"
    assert prod.forecast_invoice_count is not None
    assert any("fell back to weekday baseline" in w for w in resp.warnings)


def test_poisson_nonfinite_target_forecast_falls_back_to_weekday_baseline(
    monkeypatch,
):
    import math
    import seasonal
    from sklearn.linear_model import PoissonRegressor

    call_count = 0
    orig_predict = PoissonRegressor.predict

    def predict_mock(self, X):
        nonlocal call_count
        call_count += 1
        if call_count > 1:  # holdout predict is call 1; full target predict is call 2
            return [float("inf")] * len(X)
        return orig_predict(self, X)

    monkeypatch.setattr(PoissonRegressor, "predict", predict_mock)

    payload = sample_request_payload(history_days=100, pos_obs_days=30)
    req = SeasonalForecastRequest.model_validate(payload)
    resp = seasonal.calculate_forecast_statistics(req)

    prod = resp.products[0]
    assert prod.model_status == "WEEKDAY_BASELINE"
    assert prod.forecast_invoice_count is not None
    assert math.isfinite(prod.forecast_invoice_count)
    assert any("Poisson regression on full history failed" in w for w in resp.warnings)
