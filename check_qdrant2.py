import json
import os
import time
from collections import defaultdict
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList
from sqlalchemy import create_engine, text

load_dotenv()

DEFAULT_DOMAIN = "legal"
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "1cc_legislation")

qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT", 6333)),
)

_database_url = os.getenv("DATABASE_URL")
engine = create_engine(_database_url) if _database_url else None


def _extract_document_metadata(row: pd.Series) -> dict[str, Any]:
    raw = row.get("document_metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _resolve_document_domain(row: pd.Series, document_meta: dict[str, Any]) -> str:
    domain = row.get("domain") or document_meta.get("domain")
    return str(domain).strip() if domain else DEFAULT_DOMAIN


def _resolve_collection_name(domain: str) -> str:
    # Optional map to support domain-specific collections.
    # Example env value:
    # QDRANT_COLLECTION_MAP_JSON={"legal":"1cc_legislation","finance":"1cc_finance"}
    raw_map = os.getenv("QDRANT_COLLECTION_MAP_JSON", "").strip()
    if raw_map:
        try:
            collection_map = json.loads(raw_map)
            if isinstance(collection_map, dict) and domain in collection_map:
                return str(collection_map[domain])
        except json.JSONDecodeError:
            pass
    return DEFAULT_COLLECTION


def _resolve_country_from_row(row: pd.Series) -> str | None:
    country = row.get("country")
    if country is None:
        metadata = _extract_document_metadata(row)
        country = metadata.get("country") or metadata.get("country_code")
    if country is None:
        return None
    return str(country)


def _qdrant_scroll_document_points(
    collection_name: str,
    legislation_id: int,
    domain: str,
) -> list[Any]:
    all_points: list[Any] = []
    offset = None

    while True:
        batch, offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=legislation_id),
                    ),
                    FieldCondition(
                        key="domain",
                        match=MatchValue(value=domain),
                    ),
                ],
            ),
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(batch)
        if offset is None:
            break

    return all_points


def _build_document_dedup_ids(points: list[Any]) -> list[str]:
    seen: dict[tuple[str, str, str], str] = {}
    duplicate_ids: list[str] = []

    for point in points:
        payload = point.payload or {}
        dedup_key = (
            str(payload.get("document_id") or ""),
            str(payload.get("chunk_id") or ""),
            str(payload.get("text") or ""),
        )
        point_id = str(point.id)

        if dedup_key in seen:
            duplicate_ids.append(point_id)
        else:
            seen[dedup_key] = point_id

    return duplicate_ids


def audit_vector_presence_and_duplicates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], list[int], dict[str, list[str]]]:
    if df.empty:
        return (
            pd.DataFrame(),
            {
                "checked": 0,
                "present": 0,
                "missing": 0,
                "qdrant_errors": 0,
                "duplicate_points": 0,
            },
            [],
            {},
        )

    rows: list[dict[str, Any]] = []
    missing_ids: list[int] = []
    dedup_plan: dict[str, list[str]] = defaultdict(list)

    summary = {
        "checked": 0,
        "present": 0,
        "missing": 0,
        "qdrant_errors": 0,
        "duplicate_points": 0,
    }

    for _, row in df.iterrows():
        summary["checked"] += 1

        legislation_id_raw = row.get("legislation_id")
        if legislation_id_raw is None or (
            isinstance(legislation_id_raw, float) and pd.isna(legislation_id_raw)
        ):
            rows.append(
                {
                    "legislation_id": None,
                    "country": None,
                    "domain": None,
                    "collection_name": None,
                    "status": "invalid_legislation_id",
                    "points_found": 0,
                    "duplicate_points": 0,
                    "error": "Null legislation_id",
                },
            )
            continue

        try:
            legislation_id = int(legislation_id_raw)
        except (TypeError, ValueError):
            rows.append(
                {
                    "legislation_id": legislation_id_raw,
                    "country": None,
                    "domain": None,
                    "collection_name": None,
                    "status": "invalid_legislation_id",
                    "points_found": 0,
                    "duplicate_points": 0,
                    "error": "Non-integer legislation_id",
                },
            )
            continue

        document_meta = _extract_document_metadata(row)
        domain = _resolve_document_domain(row, document_meta)
        collection_name = _resolve_collection_name(domain)
        country = _resolve_country_from_row(row)

        try:
            points = _qdrant_scroll_document_points(
                collection_name, legislation_id, domain
            )
        except Exception as exc:
            summary["qdrant_errors"] += 1
            rows.append(
                {
                    "legislation_id": legislation_id,
                    "country": country,
                    "domain": domain,
                    "collection_name": collection_name,
                    "status": "qdrant_error",
                    "points_found": 0,
                    "duplicate_points": 0,
                    "error": str(exc),
                },
            )
            continue

        point_count = len(points)
        if point_count == 0:
            summary["missing"] += 1
            missing_ids.append(legislation_id)
            rows.append(
                {
                    "legislation_id": legislation_id,
                    "country": country,
                    "domain": domain,
                    "collection_name": collection_name,
                    "status": "missing_in_qdrant",
                    "points_found": 0,
                    "duplicate_points": 0,
                    "error": None,
                },
            )
            continue

        summary["present"] += 1
        duplicate_ids = _build_document_dedup_ids(points)
        duplicate_count = len(duplicate_ids)
        if duplicate_count > 0:
            dedup_plan[collection_name].extend(duplicate_ids)
            summary["duplicate_points"] += duplicate_count

        rows.append(
            {
                "legislation_id": legislation_id,
                "country": country,
                "domain": domain,
                "collection_name": collection_name,
                "status": "present",
                "points_found": point_count,
                "duplicate_points": duplicate_count,
                "error": None,
            },
        )

    for collection_name, ids in list(dedup_plan.items()):
        dedup_plan[collection_name] = sorted(set(ids))

    audit_df = pd.DataFrame(rows)
    missing_ids = sorted(set(missing_ids))
    return audit_df, summary, missing_ids, dict(dedup_plan)


def _update_legislation_status(
    legislation_ids: list[int],
    status: str,
    expected_status: str | None = None,
) -> int:
    if not legislation_ids:
        return 0
    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured")

    with engine.begin() as conn:
        query = text(
            """
            UPDATE legislation_craw
            SET status = :status
            WHERE legislation_id = ANY(:ids)
            """ + (" AND status = :expected_status" if expected_status else ""),
        )
        params: dict[str, Any] = {"status": status, "ids": legislation_ids}
        if expected_status:
            params["expected_status"] = expected_status
        result = conn.execute(query, params)
        return int(result.rowcount or 0)


def _delete_qdrant_points(
    collection_name: str,
    point_ids: list[str],
    batch_size: int = 256,
) -> int:
    if not point_ids:
        return 0

    deleted = 0
    for i in range(0, len(point_ids), batch_size):
        batch = point_ids[i : i + batch_size]
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=batch),
            wait=True,
        )
        deleted += len(batch)
    return deleted


def apply_corrections(
    missing_ids: list[int],
    dedup_plan: dict[str, list[str]],
    dry_run: bool = True,
    requeue_status: str = "document_metadata_generated",
) -> dict[str, int]:
    summary = {
        "missing_candidates": len(missing_ids),
        "dedup_candidates": sum(len(v) for v in dedup_plan.values()),
        "status_rows_updated": 0,
        "qdrant_points_deleted": 0,
    }

    if dry_run:
        return summary

    if missing_ids:
        summary["status_rows_updated"] = _update_legislation_status(
            legislation_ids=missing_ids,
            status=requeue_status,
            expected_status="vector_db_ingested",
        )

    for collection_name, point_ids in dedup_plan.items():
        if not point_ids:
            continue
        summary["qdrant_points_deleted"] += _delete_qdrant_points(
            collection_name,
            point_ids,
        )
        time.sleep(0.05)

    return summary
