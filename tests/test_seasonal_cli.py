import csv
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from seasonal import SeasonalForecastRequest
from tools.seasonal_cli import (
    aggregate_csv_rows,
    generate_synthetic_demo_data,
    normalize_header,
    parse_iso_date,
)


def test_normalize_header():
    assert normalize_header("Invoice_ID") == "order_id"
    assert normalize_header("bouquet_id") == "product_id"
    assert normalize_header("paid-date") == "date"
    assert normalize_header("Units") == "quantity"
    assert normalize_header("product_name") == "name"


def test_parse_iso_date():
    from datetime import date

    assert parse_iso_date("2026-03-08") == date(2026, 3, 8)
    assert parse_iso_date("2026/03/08") == date(2026, 3, 8)
    assert parse_iso_date("08-03-2026") == date(2026, 3, 8)

    with pytest.raises(ValueError):
        parse_iso_date("invalid-date-string")


def test_aggregate_csv_rows_groups_distinct_orders():
    p1 = str(uuid4())
    p2 = str(uuid4())

    rows = [
        ["order_id", "product_id", "date", "quantity", "product_name"],
        # 2 distinct orders on same date for p1
        ["ORD-101", p1, "2026-01-15", "2", "Bouquet A"],
        ["ORD-102", p1, "2026-01-15", "1", "Bouquet A"],
        # Same order ORD-102 repeated with another item of p1 -> should sum quantity to 3, invoice_count remains 2
        ["ORD-102", p1, "2026-01-15", "2", "Bouquet A"],
        # Another date for p1
        ["ORD-103", p1, "2026-01-16", "1", "Bouquet A"],
        # Product p2
        ["ORD-201", p2, "2026-01-15", "5", "Bouquet B"],
        # Corrupt row with invalid UUID (should be safely skipped)
        ["ORD-BAD", "not-a-uuid", "2026-01-15", "1", "Bad"],
        # Corrupt row with invalid date (should be safely skipped)
        ["ORD-BAD2", p1, "not-a-date", "1", "Bad"],
        # Corrupt row with negative quantity (should be safely skipped)
        ["ORD-BAD3", p1, "2026-01-15", "-5", "Bad"],
    ]

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", delete=False, suffix=".csv"
    ) as tmp:
        writer = csv.writer(tmp)
        writer.writerows(rows)
        tmp_path = Path(tmp.name)

    try:
        products, observations, min_d, max_d = aggregate_csv_rows(tmp_path)

        assert len(products) == 2
        p1_obs = [o for o in observations if o["product_id"] == p1]
        assert len(p1_obs) == 2  # 2 dates: Jan 15 and Jan 16

        jan15 = next(o for o in p1_obs if o["day"] == "2026-01-15")
        # 2 distinct orders (ORD-101 and ORD-102)
        assert jan15["invoice_count"] == 2
        # Sum of quantities (2 + 1 + 2 = 5)
        assert jan15["units"] == 5

        jan16 = next(o for o in p1_obs if o["day"] == "2026-01-16")
        assert jan16["invoice_count"] == 1
        assert jan16["units"] == 1

        p2_obs = [o for o in observations if o["product_id"] == p2]
        assert len(p2_obs) == 1
        assert p2_obs[0]["invoice_count"] == 1
        assert p2_obs[0]["units"] == 5

    finally:
        tmp_path.unlink(missing_ok=True)


def test_synthetic_demo_generation_determinism_and_labels():
    payload, raw_rows = generate_synthetic_demo_data(
        target_year=2026, target_month=3, seed=42
    )

    assert payload["contract_version"] == "seasonal-v1"
    assert payload["data_source"] == "SYNTHETIC"
    assert "_disclaimer" not in payload
    assert len(payload["products"]) == 4

    # Validate against strict request schema unmodified
    req = SeasonalForecastRequest.model_validate(payload)
    assert req.contract_version == "seasonal-v1"

    # Check raw CSV rows
    assert len(raw_rows) > 0
    first_row = raw_rows[0]
    assert "order_id" in first_row
    assert "product_id" in first_row
    assert "date" in first_row
    assert "quantity" in first_row
    assert first_row["order_id"].startswith("SYN-ORD-")


def test_cli_demo_integration_validates_unmodified(tmp_path):
    import sys
    from tools.seasonal_cli import main

    out_json = tmp_path / "demo_output.json"
    out_csv = tmp_path / "demo_output.csv"

    orig_argv = sys.argv
    try:
        sys.argv = [
            "seasonal_cli.py",
            "demo",
            "--target-year",
            "2026",
            "--target-month",
            "11",
            "--output-json",
            str(out_json),
            "--output-csv",
            str(out_csv),
        ]
        main()
    finally:
        sys.argv = orig_argv

    assert out_json.is_file()
    assert out_csv.is_file()

    with open(out_json, "r", encoding="utf-8") as f:
        content = f.read()

    req = SeasonalForecastRequest.model_validate_json(content)
    assert req.contract_version == "seasonal-v1"
    assert req.data_source == "SYNTHETIC"
    assert req.target_year == 2026
    assert req.target_month == 11
    assert len(req.products) == 4
    assert len(req.observations) > 0
