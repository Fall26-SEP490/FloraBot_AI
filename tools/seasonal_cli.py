"""FloraBot Seasonal AI CLI tool.

Handles:
- Aggregating raw sales CSV (order_id, product_id, date, quantity) into distinct invoice counts
  and units per product per day for seasonal-v1 forecast payloads.
- Generating deterministic 2-year synthetic demo datasets with explicit SYNTHETIC labels
  and no real quality claims.
"""

import argparse
import csv
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import UUID

DEMO_DISCLAIMER = (
    "SYNTHETIC DEMO DATA - NO REAL QUALITY CLAIM - FOR TESTING AND DEMONSTRATION ONLY"
)


def parse_iso_date(value: str) -> date:
    val = value.strip()
    try:
        return date.fromisoformat(val)
    except ValueError:
        # Fallback for common formats like YYYY/MM/DD or DD-MM-YYYY
        for fmt in ("%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(val, fmt).date()
            except ValueError:
                pass
        raise ValueError(f"Unrecognized date format: '{value}'. Expected YYYY-MM-DD.")


def normalize_header(header: str) -> str:
    cleaned = header.strip().lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "invoice_id": "order_id",
        "invoice": "order_id",
        "order": "order_id",
        "bouquet_id": "product_id",
        "product": "product_id",
        "paid_date": "date",
        "day": "date",
        "units": "quantity",
        "count": "quantity",
        "qty": "quantity",
        "product_name": "name",
    }
    return mapping.get(cleaned, cleaned)


def aggregate_csv_rows(
    csv_file_path: str | Path,
    max_products: int = 100,
    max_observations: int = 110000,
) -> tuple[list[dict], list[dict], date, date]:
    """Reads raw invoice CSV and aggregates distinct orders and total units per day/product."""
    path = Path(csv_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {csv_file_path}")

    # Read with utf-8-sig to handle optional BOM
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            raw_headers = next(reader)
        except StopIteration:
            raise ValueError("CSV file is empty")

        headers = [normalize_header(h) for h in raw_headers]
        required = {"order_id", "product_id", "date", "quantity"}
        missing = required - set(headers)
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. Present: {raw_headers}"
            )

        idx_order = headers.index("order_id")
        idx_product = headers.index("product_id")
        idx_date = headers.index("date")
        idx_qty = headers.index("quantity")
        idx_name = headers.index("name") if "name" in headers else None

        # Data structures for aggregation
        # (day, product_id) -> set of order_ids, total units
        grouped_obs: dict[tuple[date, UUID], tuple[set[str], int]] = {}
        products_meta: dict[UUID, dict] = {}

        min_date: date | None = None
        max_date: date | None = None
        skipped_rows = 0

        for row_num, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) <= max(idx_order, idx_product, idx_date, idx_qty):
                skipped_rows += 1
                continue

            order_str = row[idx_order].strip()
            product_str = row[idx_product].strip()
            date_str = row[idx_date].strip()
            qty_str = row[idx_qty].strip()

            if not order_str or not product_str or not date_str or not qty_str:
                skipped_rows += 1
                continue

            try:
                prod_id = UUID(product_str)
            except ValueError:
                skipped_rows += 1
                continue

            try:
                row_date = parse_iso_date(date_str)
            except ValueError:
                skipped_rows += 1
                continue

            try:
                qty = int(qty_str)
                if qty <= 0:
                    skipped_rows += 1
                    continue
            except ValueError:
                skipped_rows += 1
                continue

            if min_date is None or row_date < min_date:
                min_date = row_date
            if max_date is None or row_date > max_date:
                max_date = row_date

            prod_name = (
                row[idx_name].strip()
                if idx_name is not None and len(row) > idx_name
                else f"Product {str(prod_id)[:8]}"
            )
            if not prod_name:
                prod_name = f"Product {str(prod_id)[:8]}"

            if prod_id not in products_meta:
                products_meta[prod_id] = {
                    "product_id": str(prod_id),
                    "name": prod_name[:160],
                    "available_since": row_date.isoformat(),
                }
            else:
                curr_since = date.fromisoformat(
                    products_meta[prod_id]["available_since"]
                )
                if row_date < curr_since:
                    products_meta[prod_id]["available_since"] = row_date.isoformat()

            key = (row_date, prod_id)
            if key not in grouped_obs:
                grouped_obs[key] = (set([order_str]), qty)
            else:
                orders, total_qty = grouped_obs[key]
                orders.add(order_str)
                grouped_obs[key] = (orders, total_qty + qty)

    if not products_meta or min_date is None or max_date is None:
        raise ValueError("No valid sales observations found in CSV")

    if len(products_meta) > max_products:
        raise ValueError(
            f"Found {len(products_meta)} products, exceeding limit of {max_products}"
        )

    if len(grouped_obs) > max_observations:
        raise ValueError(
            f"Found {len(grouped_obs)} daily observations, exceeding limit of {max_observations}"
        )

    covered_days = (max_date - min_date).days + 1
    if covered_days > 1096:
        raise ValueError(
            f"History covers {covered_days} days, exceeding limit of 1096 days"
        )

    products_list = list(products_meta.values())
    observations_list = []
    for (d, pid), (orders, total_qty) in sorted(
        grouped_obs.items(), key=lambda x: (x[0][0], str(x[0][1]))
    ):
        inv_count = len(orders)
        units = max(inv_count, total_qty)  # Guarantee inv_count <= units
        observations_list.append(
            {
                "day": d.isoformat(),
                "product_id": str(pid),
                "invoice_count": inv_count,
                "units": units,
            }
        )

    return products_list, observations_list, min_date, max_date


def generate_synthetic_demo_data(
    target_year: int = 2026,
    target_month: int = 3,
    history_days: int = 730,
    seed: int = 42,
) -> tuple[dict, list[dict]]:
    """Generates a deterministic 2-year synthetic dataset for demonstration and testing.

    Includes distinct product archetypes (seasonal spike, consistent baseline, cold start, low activity).
    """
    rng = random.Random(seed)

    forecast_start = date(target_year, target_month, 1)
    # End history on the day before forecast_start
    history_end = forecast_start - timedelta(days=1)
    history_start = history_end - timedelta(days=history_days - 1)

    products = [
        {
            "product_id": "60000000-0000-0000-0000-000000000001",
            "name": "[SYNTHETIC] Hoa Hong Valentine & 8-3 (Seasonal Spike)",
            "available_since": history_start.isoformat(),
            "pattern": "seasonal",
        },
        {
            "product_id": "60000000-0000-0000-0000-000000000002",
            "name": "[SYNTHETIC] Bo Hoa Ly Trang Nha (Steady Weekday)",
            "available_since": history_start.isoformat(),
            "pattern": "steady",
        },
        {
            "product_id": "60000000-0000-0000-0000-000000000003",
            "name": "[SYNTHETIC] Lan Ho Diep Cao Cap (Cold Start - 30 days)",
            "available_since": (history_end - timedelta(days=29)).isoformat(),
            "pattern": "cold_start",
        },
        {
            "product_id": "60000000-0000-0000-0000-000000000004",
            "name": "[SYNTHETIC] Hoa Huong Duong Vang (Low Observations)",
            "available_since": history_start.isoformat(),
            "pattern": "sparse",
        },
    ]

    observations: list[dict] = []
    raw_csv_rows: list[dict] = []
    order_seq = 1

    current = history_start
    while current <= history_end:
        # Product 1: Strong seasonal spikes on Feb 7-14, Mar 1-8, Oct 13-20
        # Normal daily base: 1 to 4 invoices
        is_spikemonth = (
            (current.month == 2 and 7 <= current.day <= 14)
            or (current.month == 3 and 1 <= current.day <= 8)
            or (current.month == 10 and 13 <= current.day <= 20)
        )
        p1_inv = (
            rng.randint(8, 20)
            if is_spikemonth
            else (rng.randint(1, 4) if rng.random() > 0.3 else 0)
        )
        if p1_inv > 0:
            p1_units = p1_inv + rng.randint(0, p1_inv)
            pid1 = products[0]["product_id"]
            observations.append(
                {
                    "day": current.isoformat(),
                    "product_id": pid1,
                    "invoice_count": p1_inv,
                    "units": p1_units,
                }
            )
            for _ in range(p1_inv):
                raw_csv_rows.append(
                    {
                        "order_id": f"SYN-ORD-{order_seq:07d}",
                        "product_id": pid1,
                        "date": current.isoformat(),
                        "quantity": 1 + (1 if rng.random() > 0.6 else 0),
                        "name": products[0]["name"],
                    }
                )
                order_seq += 1

        # Product 2: Steady weekday pattern (Fri/Sat higher)
        p2_base = 3 if current.weekday() in (4, 5) else 1
        p2_inv = p2_base + (1 if rng.random() > 0.5 else 0)
        p2_units = p2_inv + (1 if rng.random() > 0.7 else 0)
        pid2 = products[1]["product_id"]
        observations.append(
            {
                "day": current.isoformat(),
                "product_id": pid2,
                "invoice_count": p2_inv,
                "units": p2_units,
            }
        )
        for _ in range(p2_inv):
            raw_csv_rows.append(
                {
                    "order_id": f"SYN-ORD-{order_seq:07d}",
                    "product_id": pid2,
                    "date": current.isoformat(),
                    "quantity": 1,
                    "name": products[1]["name"],
                }
            )
            order_seq += 1

        # Product 3: Cold start (available last 30 days only)
        if current >= (history_end - timedelta(days=29)):
            p3_inv = rng.randint(1, 2) if rng.random() > 0.5 else 0
            if p3_inv > 0:
                p3_units = p3_inv
                pid3 = products[2]["product_id"]
                observations.append(
                    {
                        "day": current.isoformat(),
                        "product_id": pid3,
                        "invoice_count": p3_inv,
                        "units": p3_units,
                    }
                )
                for _ in range(p3_inv):
                    raw_csv_rows.append(
                        {
                            "order_id": f"SYN-ORD-{order_seq:07d}",
                            "product_id": pid3,
                            "date": current.isoformat(),
                            "quantity": 1,
                            "name": products[2]["name"],
                        }
                    )
                    order_seq += 1

        # Product 4: Sparse (< 10 observations across 2 years)
        if (history_end - current).days in (50, 120, 200, 310, 450):
            pid4 = products[3]["product_id"]
            observations.append(
                {
                    "day": current.isoformat(),
                    "product_id": pid4,
                    "invoice_count": 1,
                    "units": 1,
                }
            )
            raw_csv_rows.append(
                {
                    "order_id": f"SYN-ORD-{order_seq:07d}",
                    "product_id": pid4,
                    "date": current.isoformat(),
                    "quantity": 1,
                    "name": products[3]["name"],
                }
            )
            order_seq += 1

        current += timedelta(days=1)

    clean_products = [
        {
            "product_id": p["product_id"],
            "name": p["name"],
            "available_since": p["available_since"],
        }
        for p in products
    ]

    request_payload = {
        "contract_version": "seasonal-v1",
        "history_start": history_start.isoformat(),
        "history_end": history_end.isoformat(),
        "target_year": target_year,
        "target_month": target_month,
        "data_source": "SYNTHETIC",
        "products": clean_products,
        "observations": observations,
        "include_explanations": True,
    }

    return request_payload, raw_csv_rows


def main():
    parser = argparse.ArgumentParser(
        description="FloraBot Seasonal AI Data CLI (Aggregation & Synthetic Demo)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: aggregate
    agg_parser = subparsers.add_parser(
        "aggregate", help="Aggregate raw CSV into seasonal-v1 JSON/CSV"
    )
    agg_parser.add_argument(
        "--input", required=True, help="Path to input raw sales CSV"
    )
    agg_parser.add_argument(
        "--target-year",
        type=int,
        required=True,
        help="Forecast target year (e.g. 2026)",
    )
    agg_parser.add_argument(
        "--target-month", type=int, required=True, help="Forecast target month (1-12)"
    )
    agg_parser.add_argument(
        "--data-source",
        choices=["DATABASE", "SYNTHETIC"],
        required=True,
        help="Explicit caller data source provenance (DATABASE or SYNTHETIC; required)",
    )
    agg_parser.add_argument("--output-json", help="Path to save generated JSON payload")
    agg_parser.add_argument(
        "--output-csv", help="Path to save aggregated observations CSV"
    )

    # Subcommand: demo
    demo_parser = subparsers.add_parser(
        "demo", help="Generate deterministic 2-year synthetic demo dataset"
    )
    demo_parser.add_argument(
        "--target-year", type=int, default=2026, help="Target year (default: 2026)"
    )
    demo_parser.add_argument(
        "--target-month", type=int, default=3, help="Target month (default: 3)"
    )
    demo_parser.add_argument(
        "--output-json",
        default="synthetic_request.json",
        help="Path to save JSON request",
    )
    demo_parser.add_argument(
        "--output-csv", default="synthetic_sales.csv", help="Path to save raw sales CSV"
    )

    args = parser.parse_args()

    if args.command == "aggregate":
        products, observations, min_d, max_d = aggregate_csv_rows(args.input)
        payload = {
            "contract_version": "seasonal-v1",
            "history_start": min_d.isoformat(),
            "history_end": max_d.isoformat(),
            "target_year": args.target_year,
            "target_month": args.target_month,
            "data_source": args.data_source,
            "products": products,
            "observations": observations,
            "include_explanations": True,
        }

        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"Saved aggregated JSON to {args.output_json}")

        if args.output_csv:
            with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["day", "product_id", "invoice_count", "units"])
                for obs in observations:
                    writer.writerow(
                        [
                            obs["day"],
                            obs["product_id"],
                            obs["invoice_count"],
                            obs["units"],
                        ]
                    )
            print(f"Saved aggregated CSV to {args.output_csv}")

        print(
            f"Aggregation complete: {len(products)} products, {len(observations)} observation days ({min_d} to {max_d})."
        )

    elif args.command == "demo":
        payload, raw_rows = generate_synthetic_demo_data(
            target_year=args.target_year, target_month=args.target_month
        )

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["order_id", "product_id", "date", "quantity", "name"]
            )
            writer.writeheader()
            writer.writerows(raw_rows)

        print(
            f"Generated 2-year synthetic demo data: {len(payload['products'])} products, {len(payload['observations'])} daily aggregated observations."
        )
        print(f"Disclaimer: {DEMO_DISCLAIMER}")
        print(f"Wrote JSON payload: {args.output_json}")
        print(f"Wrote raw CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
