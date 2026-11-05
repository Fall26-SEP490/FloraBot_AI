import asyncio
import calendar
import json
import math
import re
import warnings as py_warnings
from datetime import date, datetime, timedelta
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor

TIMEZONE_VN = ZoneInfo("Asia/Ho_Chi_Minh")

# Season mapping: SPRING (Mar-May), SUMMER (Jun-Aug), AUTUMN (Sep-Nov), WINTER (Dec-Feb)
MONTH_TO_SEASON: dict[int, Literal["SPRING", "SUMMER", "AUTUMN", "WINTER"]] = {
    1: "WINTER",
    2: "WINTER",
    3: "SPRING",
    4: "SPRING",
    5: "SPRING",
    6: "SUMMER",
    7: "SUMMER",
    8: "SUMMER",
    9: "AUTUMN",
    10: "AUTUMN",
    11: "AUTUMN",
    12: "WINTER",
}

SEASON_VI_NAMES: dict[str, str] = {
    "SPRING": "Mùa Xuân",
    "SUMMER": "Mùa Hạ",
    "AUTUMN": "Mùa Thu",
    "WINTER": "Mùa Đông",
}

# Gregorian flower holidays: (month, day, code, name)
GREGORIAN_HOLIDAYS: list[tuple[int, int, str, str]] = [
    (1, 1, "TET_DUONG_LICH", "Tết Dương Lịch"),
    (2, 14, "VALENTINE", "Ngày Lễ Tình Nhân"),
    (3, 8, "QUOC_TE_PHU_NU", "Quốc tế Phụ nữ"),
    (10, 20, "PHU_NU_VIET_NAM", "Phụ nữ Việt Nam"),
    (11, 20, "NHA_GIAO_VIET_NAM", "Nhà giáo Việt Nam"),
    (12, 25, "GIANG_SINH", "Lễ Giáng Sinh"),
]

DEMAND_CAVEAT = (
    "Forecasts represent observed sales/invoice frequency, not unconstrained true demand "
    "or guaranteed stocking quantity; unobserved stockouts may cause downward bias."
)

ANNUAL_SEASONALITY_WARNING = (
    "Annual seasonality is not validated (14-day holdout does not validate multi-year seasonal demand; "
    "no annual rolling backtest exists). Historical coverage available for reference only."
)

UNSAFE = re.compile(
    r"https?://|www\.|ignore|bỏ qua hướng dẫn|system prompt|<[a-z/]|"
    r"override|instructions?|developer message|\[/?INST\]|"
    r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
    re.IGNORECASE,
)

UNSUPPORTED_COMPARISON = re.compile(
    r"\btăng\b|\bgiảm\b|chu\s+kỳ\s+hàng\s+năm|"
    r"\b(?:increase|decrease|growth|decline|rise|drop)\b|"
    r"cùng\s+kỳ|"
    r"năm\s+(?:ngoái|trước|cũ|qua)|"
    r"so\s+với\s+(?:năm|cùng|mùa|tháng)|"
    r"sụt\s+giảm|"
    r"tăng\s+trưởng|"
    r"suy\s+giảm|"
    r"\d+(?:\.\d+)?\s*%|"
    r"phần\s+trăm|"
    r"year[\s\-_]*over[\s\-_]*year|\byoy\b|"
    r"last\s+year|prior\s+year|previous\s+year|"
    r"compared\s+to\s+last|versus\s+last|vs\.?\s+last|"
    r"growth\s+rate|%\s*(?:increase|decrease|drop|growth)",
    re.IGNORECASE,
)


def clean_text(value: str) -> str:
    return " ".join(
        sentence.strip()
        for sentence in re.split(r"[\n\r.!?]+", value)
        if sentence.strip() and not UNSAFE.search(sentence)
    )


def get_season_for_month(
    month: int,
) -> Literal["SPRING", "SUMMER", "AUTUMN", "WINTER"]:
    return MONTH_TO_SEASON[month]


def is_holiday_event_day(d: date) -> bool:
    return any(d.month == h_m and d.day == h_d for h_m, h_d, _, _ in GREGORIAN_HOLIDAYS)


def is_in_holiday_window(d: date) -> bool:
    # 7 preceding days plus event day (8 days total)
    for year in (d.year - 1, d.year, d.year + 1):
        for h_m, h_d, _, _ in GREGORIAN_HOLIDAYS:
            try:
                event_date = date(year, h_m, h_d)
            except ValueError:
                continue
            if event_date - timedelta(days=7) <= d <= event_date:
                return True
    return False


def get_target_month_holidays(
    target_year: int, target_month: int
) -> list["HolidayItem"]:
    month_start = date(target_year, target_month, 1)
    num_days = calendar.monthrange(target_year, target_month)[1]
    month_end = date(target_year, target_month, num_days)

    found_holidays: list[HolidayItem] = []
    seen_codes: set[str] = set()

    for year in (target_year - 1, target_year, target_year + 1):
        for h_m, h_d, code, name in GREGORIAN_HOLIDAYS:
            try:
                event_date = date(year, h_m, h_d)
            except ValueError:
                continue
            window_start = event_date - timedelta(days=7)
            window_end = event_date
            if (
                window_start <= month_end
                and window_end >= month_start
                and code not in seen_codes
            ):
                seen_codes.add(code)
                found_holidays.append(HolidayItem(code=code, name=name, day=event_date))

    found_holidays.sort(key=lambda h: h.day)
    return found_holidays


def get_current_vietnam_calendar_context() -> str:
    now_vn = datetime.now(TIMEZONE_VN).date()
    season = get_season_for_month(now_vn.month)
    season_vi = SEASON_VI_NAMES.get(season, season)

    active_holidays: list[str] = []
    for h_m, h_d, _, name in GREGORIAN_HOLIDAYS:
        try:
            event_date = date(now_vn.year, h_m, h_d)
        except ValueError:
            continue
        if event_date - timedelta(days=7) <= now_vn <= event_date:
            active_holidays.append(f"{name} ({h_d:02d}/{h_m:02d})")

    context = (
        f"Thời gian hiện tại tại Việt Nam: ngày {now_vn.strftime('%d/%m/%Y')}, "
        f"thuộc {season_vi} ({season})."
    )
    if active_holidays:
        context += f" Đang trong dịp cao điểm hoa: {', '.join(active_holidays)}."
    return context


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProductInput(StrictModel):
    product_id: UUID
    name: str = Field(min_length=1, max_length=160)
    available_since: date


class DailyObservation(StrictModel):
    day: date
    product_id: UUID
    invoice_count: int = Field(ge=0)
    units: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.invoice_count > self.units:
            raise ValueError("invoice_count cannot exceed units")
        return self


class SeasonalForecastRequest(StrictModel):
    contract_version: Literal["seasonal-v1"]
    history_start: date
    history_end: date
    target_year: int = Field(ge=2000, le=2100)
    target_month: int = Field(ge=1, le=12)
    data_source: Literal["DATABASE", "SYNTHETIC"]
    products: list[ProductInput] = Field(min_length=1, max_length=100)
    observations: list[DailyObservation] = Field(max_length=110000)
    include_explanations: bool = True

    @model_validator(mode="after")
    def validate_contract(self):
        if self.history_start > self.history_end:
            raise ValueError("history_start must be <= history_end")

        covered_days = (self.history_end - self.history_start).days + 1
        if covered_days > 1096:
            raise ValueError("history coverage cannot exceed 1096 days")

        target_start = date(self.target_year, self.target_month, 1)
        if target_start <= self.history_end:
            raise ValueError(
                "target forecast month must strictly follow history_end (no future training)"
            )

        gap_days = (target_start - self.history_end).days
        if gap_days > 366:
            raise ValueError(
                "gap between history_end and forecast_start must be <= 366 days"
            )

        product_map = {p.product_id: p for p in self.products}
        if len(product_map) != len(self.products):
            raise ValueError("duplicate product_id in products list")

        seen_obs: set[tuple[date, UUID]] = set()
        for obs in self.observations:
            if obs.product_id not in product_map:
                raise ValueError(
                    f"observation references unknown product_id {obs.product_id}"
                )
            # Strict rejection: observation cannot precede product.available_since
            prod = product_map[obs.product_id]
            if obs.day < prod.available_since:
                raise ValueError(
                    f"observation date {obs.day} precedes product available_since {prod.available_since} "
                    f"for product {obs.product_id}"
                )
            if obs.day < self.history_start or obs.day > self.history_end:
                raise ValueError("observation day out of declared history bounds")
            obs_key = (obs.day, obs.product_id)
            if obs_key in seen_obs:
                raise ValueError(
                    f"duplicate observation for date {obs.day} and product {obs.product_id}"
                )
            seen_obs.add(obs_key)

        return self


class HolidayItem(StrictModel):
    code: str
    name: str
    day: date


class ValidationMetrics(StrictModel):
    training_days: int
    holdout_days: int
    mae: float | None
    baseline_mae: float | None
    wape: float | None
    baseline_wape: float | None


class SeasonalProductForecast(StrictModel):
    product_id: UUID
    name: str
    observed_invoice_count: int
    observed_units: int
    same_season_invoice_count: int
    forecast_invoice_count: float | None
    forecast_units: float | None
    model_status: Literal[
        "INSUFFICIENT_HISTORY", "POISSON_REGRESSION", "WEEKDAY_BASELINE"
    ]
    annual_seasonality_validated: bool
    validation: ValidationMetrics
    explanation: str
    explanation_source: Literal["GEMINI", "FALLBACK"]


class SeasonalForecastResponse(StrictModel):
    contract_version: Literal["seasonal-v1"] = "seasonal-v1"
    aggregation_level: Literal["PRODUCT"] = "PRODUCT"
    data_source: Literal["DATABASE", "SYNTHETIC"]
    history_start: date
    history_end: date
    forecast_start: date
    forecast_end: date
    season: Literal["SPRING", "SUMMER", "AUTUMN", "WINTER"]
    holidays: list[HolidayItem]
    warnings: list[str]
    products: list[SeasonalProductForecast]


def extract_date_features(d: date) -> list[float]:
    # 7 weekday indicators
    weekday_flags = [1.0 if d.weekday() == w else 0.0 for w in range(7)]

    # Month cyclic (sin/cos)
    month_rad = 2.0 * math.pi * d.month / 12.0
    sin_month = math.sin(month_rad)
    cos_month = math.cos(month_rad)

    # Day of year cyclic (sin/cos)
    yday = d.timetuple().tm_yday
    yday_rad = 2.0 * math.pi * yday / 365.25
    sin_yday = math.sin(yday_rad)
    cos_yday = math.cos(yday_rad)

    # Proximity to flower holidays
    hol_window = 1.0 if is_in_holiday_window(d) else 0.0
    hol_event = 1.0 if is_holiday_event_day(d) else 0.0

    return [
        *weekday_flags,
        sin_month,
        cos_month,
        sin_yday,
        cos_yday,
        hol_window,
        hol_event,
    ]


def fit_poisson_regressor(X: list[list[float]], y: list[float]) -> PoissonRegressor:
    model = PoissonRegressor(alpha=1.0, max_iter=300, tol=1e-4)
    with py_warnings.catch_warnings(record=True) as caught_warns:
        py_warnings.simplefilter("always")
        model.fit(X, y)
        for w in caught_warns:
            if issubclass(w.category, ConvergenceWarning):
                raise ValueError("Poisson regression failed to converge")
    return model


def compute_weekday_means(days: list[date], y: list[float]) -> dict[int, float]:
    sums: dict[int, float] = {w: 0.0 for w in range(7)}
    counts: dict[int, int] = {w: 0 for w in range(7)}
    for d, val in zip(days, y):
        w = d.weekday()
        sums[w] += val
        counts[w] += 1
    return {w: (sums[w] / counts[w] if counts[w] > 0 else 0.0) for w in range(7)}


def calculate_forecast_statistics(
    payload: SeasonalForecastRequest,
) -> SeasonalForecastResponse:
    target_season = get_season_for_month(payload.target_month)
    target_holidays = get_target_month_holidays(
        payload.target_year, payload.target_month
    )

    num_days = calendar.monthrange(payload.target_year, payload.target_month)[1]
    forecast_start = date(payload.target_year, payload.target_month, 1)
    forecast_end = date(payload.target_year, payload.target_month, num_days)
    target_days = [
        date(payload.target_year, payload.target_month, d)
        for d in range(1, num_days + 1)
    ]
    target_features = [extract_date_features(d) for d in target_days]

    obs_by_product: dict[UUID, dict[date, tuple[int, int]]] = {
        p.product_id: {} for p in payload.products
    }
    for obs in payload.observations:
        obs_by_product[obs.product_id][obs.day] = (obs.invoice_count, obs.units)

    forecast_products: list[SeasonalProductForecast] = []
    warnings: list[str] = [DEMAND_CAVEAT, ANNUAL_SEASONALITY_WARNING]
    has_insufficient_history = False

    for product in payload.products:
        p_obs = obs_by_product[product.product_id]
        observed_invoice_count = sum(inv for inv, _ in p_obs.values())
        observed_units = sum(units for _, units in p_obs.values())

        same_season_invoice_count = sum(
            inv
            for d, (inv, _) in p_obs.items()
            if get_season_for_month(d.month) == target_season
        )

        effective_start = max(payload.history_start, product.available_since)
        if effective_start > payload.history_end:
            total_covered_days = 0
            covered_dates: list[date] = []
        else:
            total_covered_days = (payload.history_end - effective_start).days + 1
            covered_dates = [
                effective_start + timedelta(days=i) for i in range(total_covered_days)
            ]

        # In accordance with specification: 14-day holdout does not validate annual seasonality.
        # Field is frozen in wire contract and kept false; history cycles are annotated.
        annual_seasonality_validated = False
        annual_cycles = round(total_covered_days / 365.25, 1)

        # Zero-fill strictly over declared covered days
        y_covered: list[float] = []
        pos_invoice_days = 0
        for d in covered_dates:
            inv, _ = p_obs.get(d, (0, 0))
            y_covered.append(float(inv))
            if inv > 0:
                pos_invoice_days += 1

        is_eligible = (total_covered_days >= 70) and (pos_invoice_days >= 10)

        if not is_eligible:
            has_insufficient_history = True
            fallback_explanation = (
                f"Sản phẩm có {total_covered_days} ngày dữ liệu ({annual_cycles} chu kỳ năm) và {pos_invoice_days} ngày có đơn. "
                "Chưa đạt ngưỡng tối thiểu 70 ngày bao phủ và 10 ngày có đơn để đưa ra dự báo định lượng."
            )
            forecast_products.append(
                SeasonalProductForecast(
                    product_id=product.product_id,
                    name=product.name,
                    observed_invoice_count=observed_invoice_count,
                    observed_units=observed_units,
                    same_season_invoice_count=same_season_invoice_count,
                    forecast_invoice_count=None,
                    forecast_units=None,
                    model_status="INSUFFICIENT_HISTORY",
                    annual_seasonality_validated=annual_seasonality_validated,
                    validation=ValidationMetrics(
                        training_days=total_covered_days,
                        holdout_days=0,
                        mae=None,
                        baseline_mae=None,
                        wape=None,
                        baseline_wape=None,
                    ),
                    explanation=fallback_explanation,
                    explanation_source="FALLBACK",
                )
            )
            continue

        # Strict chronological split: last 14 days as holdout
        holdout_len = 14
        train_len = total_covered_days - holdout_len
        train_dates = covered_dates[:train_len]
        holdout_dates = covered_dates[train_len:]

        y_train = y_covered[:train_len]
        y_holdout = y_covered[train_len:]

        X_train = [extract_date_features(d) for d in train_dates]
        X_holdout = [extract_date_features(d) for d in holdout_dates]

        # Evaluate PoissonRegressor from scikit-learn
        poisson_fit_failed = False
        poisson_mae: float | None = None
        poisson_wape: float | None = None
        try:
            poisson_model = fit_poisson_regressor(X_train, y_train)
            raw_poisson_preds = poisson_model.predict(X_holdout)
            if any(not math.isfinite(p) for p in raw_poisson_preds):
                raise ValueError("Non-finite prediction encountered")
            pred_holdout_poisson = [max(0.0, float(p)) for p in raw_poisson_preds]
            poisson_mae = float(
                sum(abs(y - p) for y, p in zip(y_holdout, pred_holdout_poisson))
                / holdout_len
            )
            if not math.isfinite(poisson_mae):
                raise ValueError("Non-finite MAE encountered")
            sum_holdout = sum(y_holdout)
            poisson_wape = (
                float(
                    sum(abs(y - p) for y, p in zip(y_holdout, pred_holdout_poisson))
                    / sum_holdout
                )
                if sum_holdout > 0
                else None
            )
            if poisson_wape is not None and not math.isfinite(poisson_wape):
                poisson_wape = None
        except Exception:
            poisson_fit_failed = True
            poisson_mae = None
            poisson_wape = None
            warnings.append(
                f"Poisson regression did not converge or encountered numerical failure on holdout for product {product.name}; fell back to weekday baseline."
            )

        # Evaluate weekday trailing mean baseline on holdout
        train_weekday_means = compute_weekday_means(train_dates, y_train)
        pred_holdout_baseline = [
            train_weekday_means[d.weekday()] for d in holdout_dates
        ]
        baseline_mae = float(
            sum(abs(y - p) for y, p in zip(y_holdout, pred_holdout_baseline))
            / holdout_len
        )
        sum_holdout = sum(y_holdout)
        baseline_wape = (
            float(
                sum(abs(y - p) for y, p in zip(y_holdout, pred_holdout_baseline))
                / sum_holdout
            )
            if sum_holdout > 0
            else None
        )

        # Fallback to WEEKDAY_BASELINE if Poisson failed, produced non-finite metrics, or if baseline outperforms Poisson
        if (
            poisson_fit_failed
            or poisson_mae is None
            or not math.isfinite(poisson_mae)
            or not math.isfinite(baseline_mae)
            or baseline_mae < poisson_mae
        ):
            selected_model = "WEEKDAY_BASELINE"
        else:
            selected_model = "POISSON_REGRESSION"

        # Final forecast for target month
        if selected_model == "POISSON_REGRESSION":
            X_full = [extract_date_features(d) for d in covered_dates]
            try:
                full_model = fit_poisson_regressor(X_full, y_covered)
                raw_full_preds = full_model.predict(target_features)
                if any(not math.isfinite(p) for p in raw_full_preds):
                    raise ValueError("Non-finite forecast encountered")
                daily_forecasts = [max(0.0, float(p)) for p in raw_full_preds]
            except Exception:
                selected_model = "WEEKDAY_BASELINE"
                full_weekday_means = compute_weekday_means(covered_dates, y_covered)
                daily_forecasts = [full_weekday_means[d.weekday()] for d in target_days]
                warnings.append(
                    f"Poisson regression on full history failed for product {product.name}; fell back to weekday baseline."
                )
        else:
            full_weekday_means = compute_weekday_means(covered_dates, y_covered)
            daily_forecasts = [full_weekday_means[d.weekday()] for d in target_days]

        # Guarantee non-negative daily forecasts
        daily_forecasts = [max(0.0, float(p)) for p in daily_forecasts]
        forecast_inv = round(float(sum(daily_forecasts)), 2)
        units_per_invoice = (
            (observed_units / observed_invoice_count)
            if observed_invoice_count > 0
            else 1.0
        )
        forecast_units = round(forecast_inv * units_per_invoice, 2)

        mae_val = (
            round(poisson_mae, 4)
            if (poisson_mae is not None and math.isfinite(poisson_mae))
            else None
        )
        b_mae_val = round(baseline_mae, 4) if math.isfinite(baseline_mae) else None
        wape_val = (
            round(poisson_wape, 4)
            if (poisson_wape is not None and math.isfinite(poisson_wape))
            else None
        )
        b_wape_val = (
            round(baseline_wape, 4)
            if (baseline_wape is not None and math.isfinite(baseline_wape))
            else None
        )

        if selected_model == "POISSON_REGRESSION":
            mae_str = f"{mae_val:.2f}" if mae_val is not None else "N/A"
            b_mae_str = f"{b_mae_val:.2f}" if b_mae_val is not None else "N/A"
            fallback_explanation = (
                f"Dự báo tháng {payload.target_month}/{payload.target_year} đạt {forecast_inv:.1f} lượt đơn "
                f"({forecast_units:.1f} sản phẩm) bằng hồi quy Poisson scikit-learn (MAE kiểm thử {mae_str} so với mốc cơ sở {b_mae_str}). "
                f"Dữ liệu cùng mùa ({target_season}) ghi nhận {same_season_invoice_count} lượt đơn ({annual_cycles} chu kỳ năm)."
            )
        else:
            b_mae_str = f"{b_mae_val:.2f}" if b_mae_val is not None else "N/A"
            mae_str = (
                f"{mae_val:.2f}" if mae_val is not None else "thất bại/không hội tụ"
            )
            fallback_explanation = (
                f"Dự báo tháng {payload.target_month}/{payload.target_year} đạt {forecast_inv:.1f} lượt đơn "
                f"({forecast_units:.1f} sản phẩm) theo mô hình ngày trong tuần (MAE kiểm thử {b_mae_str} "
                f"tối ưu hơn hoặc do Poisson {mae_str}). Dữ liệu cùng mùa ({target_season}) ghi nhận {same_season_invoice_count} lượt đơn."
            )

        forecast_products.append(
            SeasonalProductForecast(
                product_id=product.product_id,
                name=product.name,
                observed_invoice_count=observed_invoice_count,
                observed_units=observed_units,
                same_season_invoice_count=same_season_invoice_count,
                forecast_invoice_count=forecast_inv,
                forecast_units=forecast_units,
                model_status=selected_model,
                annual_seasonality_validated=annual_seasonality_validated,
                validation=ValidationMetrics(
                    training_days=train_len,
                    holdout_days=holdout_len,
                    mae=mae_val,
                    baseline_mae=b_mae_val,
                    wape=wape_val,
                    baseline_wape=b_wape_val,
                ),
                explanation=fallback_explanation,
                explanation_source="FALLBACK",
            )
        )

    forecast_products.sort(
        key=lambda p: (
            -1.0 if p.forecast_invoice_count is None else p.forecast_invoice_count,
            p.observed_invoice_count,
            str(p.product_id),
        ),
        reverse=True,
    )

    if has_insufficient_history:
        warnings.append(
            "Some products have insufficient history (< 70 covered days or < 10 invoice observations) and return null forecasts."
        )

    return SeasonalForecastResponse(
        contract_version="seasonal-v1",
        aggregation_level="PRODUCT",
        data_source=payload.data_source,
        history_start=payload.history_start,
        history_end=payload.history_end,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        season=target_season,
        holidays=target_holidays,
        warnings=warnings,
        products=forecast_products,
    )


SELLER_EXPLANATION_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI phân tích dự báo bán hàng hoa cho người bán (seller) tại Việt Nam. "
    "Dữ liệu được cung cấp là thống kê và dự báo đã chốt, không được tự ý sửa đổi số liệu, "
    "không thay đổi thứ tự hay thêm bớt sản phẩm. "
    "LƯU Ý NGHIÊM NGẶT VỀ THỰC TẾ DỮ LIỆU: "
    "1. Dữ liệu lịch sử là số đơn bán quan sát được (observed sales), không đại diện cho toàn bộ nhu cầu thực tế (not true demand) do có thể thiếu hàng chưa quan sát được. "
    "2. Tính mùa vụ theo năm chưa được kiểm chứng khoa học (annual seasonality unvalidated). "
    "3. TUYỆT ĐỐI KHÔNG so sánh cùng kỳ năm trước / năm ngoái (YoY / Year-over-Year), không so sánh năm hiện tại với năm trước, không đưa ra tỷ lệ phần trăm tăng trưởng hoặc sụt giảm (%) vì không có dữ liệu đối sánh cùng kỳ theo năm. "
    "4. TUYỆT ĐỐI KHÔNG so sánh tổng tích lũy toàn mùa (same_season_invoices) với dự báo của một tháng đơn lẻ (forecast_invoices) như là mức tăng hoặc sụt giảm. "
    "Chỉ giải thích ngắn gọn, chuyên nghiệp bằng tiếng Việt (tối đa 200 ký tự mỗi sản phẩm) "
    "cho các product_id được yêu cầu, dựa trên đặc điểm sản phẩm, dịp lễ trong tháng dự báo và phân bổ ngày trong tuần. "
    "Không chứa đường dẫn, HTML, cam kết tuyệt đối hay hướng dẫn can thiệp hệ thống. "
    "Chỉ trả JSON theo schema được định nghĩa."
)

SELLER_EXPLANATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "explanations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "product_id": {"type": "STRING"},
                    "explanation": {"type": "STRING"},
                },
                "required": ["product_id", "explanation"],
            },
        }
    },
    "required": ["explanations"],
}


async def enrich_with_gemini_explanations(
    response: SeasonalForecastResponse,
    settings,
    client: httpx.AsyncClient,
) -> None:
    if not response.products:
        return

    top_candidates = response.products[:10]
    candidate_id_set = {str(p.product_id) for p in top_candidates}

    prompt_data = {
        "target_season": response.season,
        "target_year": response.forecast_start.year,
        "target_month": response.forecast_start.month,
        "holidays": [h.name for h in response.holidays],
        "products": [
            {
                "product_id": str(p.product_id),
                "name": clean_text(p.name),
                "observed_invoices": p.observed_invoice_count,
                "observed_units": p.observed_units,
                "same_season_invoices": p.same_season_invoice_count,
                "forecast_invoices": p.forecast_invoice_count,
                "forecast_units": p.forecast_units,
                "model_status": p.model_status,
                "annual_seasonality_validated": p.annual_seasonality_validated,
            }
            for p in top_candidates
        ],
    }

    body = {
        "systemInstruction": {"parts": [{"text": SELLER_EXPLANATION_SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": json.dumps(prompt_data, ensure_ascii=False)}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
            "responseSchema": SELLER_EXPLANATION_SCHEMA,
        },
    }

    # Bounded to at most 8.0s per specification
    deadline = min(getattr(settings, "seller_deadline_seconds", 8.0), 8.0)

    try:
        async with asyncio.timeout(deadline):
            async with client.stream(
                "POST",
                f"https://generativelanguage.googleapis.com/v1beta/models/{settings.model}:generateContent",
                headers={"x-goog-api-key": settings.api_key},
                json=body,
            ) as res:
                res.raise_for_status()
                raw = bytearray()
                async for chunk in res.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 32768:
                        return

        envelope = json.loads(raw)
        candidate = envelope["candidates"][0]
        if not isinstance(candidate, dict) or candidate.get("finishReason") != "STOP":
            return

        text = "".join(part["text"] for part in candidate["content"]["parts"])
        data = json.loads(text)
        explanations = data.get("explanations", [])
        if not isinstance(explanations, list):
            return

        explanation_map: dict[str, str] = {}
        for item in explanations:
            if not isinstance(item, dict):
                continue
            pid = item.get("product_id")
            expl = item.get("explanation")
            if (
                pid
                and expl
                and isinstance(pid, str)
                and isinstance(expl, str)
                and pid in candidate_id_set
            ):
                cleaned = clean_text(expl)
                if (
                    cleaned
                    and not UNSAFE.search(cleaned)
                    and not UNSUPPORTED_COMPARISON.search(cleaned)
                ):
                    explanation_map[pid] = cleaned[:300]

        for p in top_candidates:
            pid_str = str(p.product_id)
            if pid_str in explanation_map:
                p.explanation = explanation_map[pid_str]
                p.explanation_source = "GEMINI"

    except Exception:
        pass
