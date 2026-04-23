"""
Mirror channels from a parent channel to a new channel ID.

Reads a CSV with columns: Parent Channel ID, Channel ID, New Channel
For each row:
  1. Reads the parent channel from Firestore
  2. Generates fresh AI enrichment for the new channel identity
  3. Writes/updates the mirrored channel document in Firestore
  4. Clones/updates BigQuery embedding rows for the new channel

Usage:
  uv run python scripts/mirror.py mirror.csv
  uv run python scripts/mirror.py mirror.csv --dry-run
"""

import csv
import os
import json
import copy
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from google.cloud import firestore
from google.cloud import bigquery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI client imports
try:
    from .open_ai_client import ChannelInsightsClient, ChannelSemanticClient
except ImportError:
    from open_ai_client import ChannelInsightsClient, ChannelSemanticClient


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

SUMMARY_INSTRUCTIONS = """
Using the metadata above and any reliable public knowledge, write a structured JSON object describing this YouTube channel.
Use this template exactly, filling in all fields where possible:

{
  "overview": "...",
  "content_topics": ["..."],
  "signature_series": ["..."],
  "target_audience": "...",
  "tone_style": "...",
  "keywords": ["..."],
  "video_success_factors": ["..."],
  "value_proposition": "..."
}

Important:
- Exclude any content that is adult, violent, spammy, misleading, or inappropriate for general audiences.
- Only include topics, keywords, or series that are verifiable or clearly relevant to the channel.
- Keep all text factual and suitable for semantic search.
"""

AI_TEMPLATE = {
    "overview": None,
    "content_topics": [],
    "signature_series": [],
    "target_audience": None,
    "tone_style": None,
    "keywords": [],
    "video_success_factors": [],
    "value_proposition": None,
}

# Ensure Google Application Credentials are set and valid
current_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not current_creds or not os.path.exists(current_creds):
    # If not in env or file doesn't exist at that path, look for service.json in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    service_json_path = os.path.join(script_dir, "service.json")
    if os.path.exists(service_json_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_json_path

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def log(level: str, msg: str, report: List[Dict]) -> None:
    ts = datetime.utcnow().isoformat()
    print(f"[{ts}] {level} {msg}")
    report.append({"ts": ts, "level": level, "message": msg})


def json_serializable(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def bq_rows_are_equal(
    bq_client: bigquery.Client,
    table_id: str,
    id_field: str,
    parent_id: str,
    new_channel_id: str,
    report: List[Dict],
    row_num: int,
    name_field: Optional[str] = None,
    new_channel_name: Optional[str] = None,
) -> bool:
    """
    Compare parent rows vs target rows fully inside BigQuery.
    Works with ARRAY fields by comparing JSON-stringified rows.
    """
    if name_field:
        parent_norm_sql = f"""
            SELECT TO_JSON_STRING(t) AS row_json
            FROM (
                SELECT * REPLACE(
                    @new_channel_id AS {id_field},
                    @new_channel_name AS {name_field}
                )
                FROM `{table_id}`
                WHERE {id_field} = @parent_id
            ) AS t
        """
        params = [
            bigquery.ScalarQueryParameter("parent_id", "STRING", parent_id),
            bigquery.ScalarQueryParameter("new_channel_id", "STRING", new_channel_id),
            bigquery.ScalarQueryParameter("new_channel_name", "STRING", new_channel_name or ""),
        ]
    else:
        parent_norm_sql = f"""
            SELECT TO_JSON_STRING(t) AS row_json
            FROM (
                SELECT * REPLACE(
                    @new_channel_id AS {id_field}
                )
                FROM `{table_id}`
                WHERE {id_field} = @parent_id
            ) AS t
        """
        params = [
            bigquery.ScalarQueryParameter("parent_id", "STRING", parent_id),
            bigquery.ScalarQueryParameter("new_channel_id", "STRING", new_channel_id),
        ]

    compare_sql = f"""
    WITH parent_norm AS (
        {parent_norm_sql}
    ),
    target_rows AS (
        SELECT TO_JSON_STRING(t) AS row_json
        FROM (
            SELECT *
            FROM `{table_id}`
            WHERE {id_field} = @new_channel_id
        ) AS t
    )
    SELECT
      (SELECT COUNT(*) FROM (
          SELECT row_json FROM parent_norm
          EXCEPT DISTINCT
          SELECT row_json FROM target_rows
      )) AS missing_in_target,
      (SELECT COUNT(*) FROM (
          SELECT row_json FROM target_rows
          EXCEPT DISTINCT
          SELECT row_json FROM parent_norm
      )) AS extra_in_target
    """

    try:
        result = list(
            bq_client.query(
                compare_sql,
                job_config=bigquery.QueryJobConfig(query_parameters=params),
            ).result(timeout=120)
        )[0]

        same = result["missing_in_target"] == 0 and result["extra_in_target"] == 0

        if same:
            log(
                "ℹ️",
                f"Row {row_num}: BQ '{table_id.split('.')[-1]}' already up to date — skipping",
                report,
            )

        return same

    except Exception as exc:
        log(
            "⚠️",
            f"Row {row_num}: BQ comparison query failed ({exc}) - assuming diff",
            report,
        )
        return False


def sync_bq_table_server_side(
    bq_client: bigquery.Client,
    table_id: str,
    id_field: str,
    parent_id: str,
    new_channel_id: str,
    report: List[Dict],
    row_num: int,
    name_field: Optional[str] = None,
    new_channel_name: Optional[str] = None,
) -> None:
    """
    Replace target rows using server-side BigQuery SQL only.
    No large row transfer to Python.
    """
    if name_field:
        insert_select = f"""
            SELECT * REPLACE(@new_channel_id AS {id_field}, @new_channel_name AS {name_field})
            FROM `{table_id}`
            WHERE {id_field} = @parent_id
        """
        params = [
            bigquery.ScalarQueryParameter("parent_id", "STRING", parent_id),
            bigquery.ScalarQueryParameter("new_channel_id", "STRING", new_channel_id),
            bigquery.ScalarQueryParameter("new_channel_name", "STRING", new_channel_name or ""),
        ]
    else:
        insert_select = f"""
            SELECT * REPLACE(@new_channel_id AS {id_field})
            FROM `{table_id}`
            WHERE {id_field} = @parent_id
        """
        params = [
            bigquery.ScalarQueryParameter("parent_id", "STRING", parent_id),
            bigquery.ScalarQueryParameter("new_channel_id", "STRING", new_channel_id),
        ]

    sql = f"""
    BEGIN TRANSACTION;

    DELETE FROM `{table_id}`
    WHERE {id_field} = @new_channel_id;

    INSERT INTO `{table_id}`
    {insert_select};

    COMMIT TRANSACTION;
    """

    try:
        bq_client.query(
            sql,
            job_config=bigquery.QueryJobConfig(query_parameters=params),
        ).result(timeout=180)
    except Exception as sync_err:
        err_msg = str(sync_err)
        is_buffer = "streaming buffer" in err_msg.lower() or "400" in err_msg
        
        if is_buffer:
             log(
                "⚠️",
                f"Row {row_num}: BQ Sync deferred — '{table_id.split('.')[-1]}' rows are in streaming buffer.",
                report,
            )
             return
        else:
            raise sync_err

    # Quick verify count
    count_sql = f"SELECT COUNT(*) AS cnt FROM `{table_id}` WHERE {id_field} = @new_channel_id"
    count_result = list(
        bq_client.query(
            count_sql,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("new_channel_id", "STRING", new_channel_id)
                ]
            ),
        ).result(timeout=60)
    )[0]

    log(
        "✅",
        f"Row {row_num}: synced {count_result['cnt']} rows → {table_id.split('.')[-1]}",
        report,
    )


def build_openai_context(metadata: Dict[str, Any], new_channel_name: str) -> str:
    context_lines: List[str] = [
        f"You are describing a NEW YouTube channel called '{new_channel_name}'.",
        "The following metadata is from the parent channel and should be used as "
        "reference context only. Tailor the description to the new channel identity.",
        "",
        "Parent channel metadata from YouTube Data API:",
    ]

    def add_line(label: str, value: Optional[Any]) -> None:
        if value is None:
            return
        if isinstance(value, list):
            if not value:
                return
            context_lines.append(f"{label}: {', '.join(str(item) for item in value)}")
        else:
            stripped = str(value).strip()
            if stripped:
                context_lines.append(f"{label}: {stripped}")

    add_line("Channel ID", metadata.get("channel_id"))
    add_line("Title", metadata.get("title"))
    add_line("Subtitle", metadata.get("subtitle"))
    add_line("Description", metadata.get("description"))
    add_line("Channel URL", metadata.get("channel_url"))
    add_line("Custom URL", metadata.get("custom_url"))
    add_line("Default Language", metadata.get("default_language"))
    add_line("Country", metadata.get("country"))
    add_line("Published At", metadata.get("published_at"))
    add_line("Keywords", metadata.get("tags"))
    add_line("Topic Categories", metadata.get("topic_categories"))

    return "\n".join(context_lines)


def validate_env() -> None:
    fs_project = os.getenv("FIRESTORE_PROJECT_ID", "")
    bq_project = os.getenv("GOOGLE_PROJECT_ID", "")

    if fs_project and bq_project:
        fs_is_prod = "prod" in fs_project.lower()
        bq_is_prod = "prod" in bq_project.lower()
        if fs_is_prod != bq_is_prod:
            raise ValueError(
                f"Environment mismatch — Firestore targets '{fs_project}' "
                f"but BigQuery targets '{bq_project}'. "
                "Both must be either prod or dev. Fix your .env file."
            )


def derive_custom_url(url_or_handle: str) -> Optional[str]:
    """
    Return a handle/path-style custom_url, not a full URL.
    Examples:
      https://www.youtube.com/@timelinechannel -> @timelinechannel
      https://youtube.com/c/SomeChannel -> c/SomeChannel
      @timelinechannel -> @timelinechannel
    """
    if not url_or_handle:
        return None

    raw = url_or_handle.strip()
    if not raw:
        return None

    if raw.startswith("@"):
        return raw

    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        path = (parsed.path or "").strip("/")
        if path:
            return path
        return None

    # Already looks like a path-like custom URL, not a full URL
    return raw.strip("/")


# -------------------------------------------------------------------
# Main Logic
# -------------------------------------------------------------------

def mirror_channels(csv_file: str, dry_run: bool = False) -> None:
    validate_env()

    db = firestore.Client(
        database=os.getenv("FIRESTORE_DATABASE", "lds-dtp-krowten-masterdata-prod-v2"),
        project=os.getenv("FIRESTORE_PROJECT_ID", "lds-krowten-prod"),
    )

    bq_project = os.getenv("GOOGLE_PROJECT_ID", "lds-krowten-prod")
    bq_dataset = os.getenv("BQ_DATASET", "krowten")
    bq_client = bigquery.Client(project=bq_project)

    source_col = db.collection("channels")
    target_col = db.collection("channels")

    report: List[Dict] = []
    insights_client: Optional[ChannelInsightsClient] = None
    semantic_client: Optional[ChannelSemanticClient] = None

    try:
        insights_client = ChannelInsightsClient(template=AI_TEMPLATE)
        semantic_client = ChannelSemanticClient(insights_client=insights_client)

        seen_targets: Set[str] = set()
        count = 0

        with open(csv_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)

            for row_num, row in enumerate(reader, start=1):
                parent_id = row.get("Parent Channel ID", "").strip()
                new_channel_id = row.get("Channel ID", "").strip()
                new_channel_name = row.get("New Channel", "").strip()
                new_channel_url = row.get("Copycat URL 1", "").strip() or row.get("Copycat URL 2", "").strip()

                if not parent_id or not new_channel_id:
                    log("⚠️", f"Row {row_num}: missing IDs — skipped", report)
                    continue

                if not new_channel_name:
                    log("⚠️", f"Row {row_num}: missing channel name — skipped", report)
                    continue

                if new_channel_id in seen_targets:
                    log("❌", f"Row {row_num}: duplicate target '{new_channel_id}' in CSV — aborting", report)
                    raise ValueError(f"Duplicate target channel ID in CSV: {new_channel_id}")
                seen_targets.add(new_channel_id)

                parent_doc = source_col.document(parent_id).get()
                if not parent_doc.exists:
                    log("❌", f"Row {row_num}: parent '{parent_id}' not found — skipped", report)
                    continue

                original_data = parent_doc.to_dict()
                channel_metadata = copy.deepcopy(original_data.get("channel_metadata", {}))

                # Update channel metadata explicitly for the new mirrored identity
                channel_metadata["title"] = new_channel_name
                if new_channel_url:
                    channel_metadata["channel_url"] = new_channel_url
                    derived_custom = derive_custom_url(new_channel_url)
                    if derived_custom:
                        channel_metadata["custom_url"] = derived_custom

                target_ref = target_col.document(new_channel_id)
                target_doc = target_ref.get()
                target_existed = target_doc.exists
                target_snapshot = target_doc.to_dict() if target_existed else None

                flags = copy.deepcopy(original_data.get("flags", {}))
                flags["is-external"] = False
                flags["is-mirrored"] = True

                log("🤖", f"Row {row_num}: generating AI enrichment for '{new_channel_name}'...", report)
                ai_status = "success"
                try:
                    context = f"{build_openai_context(channel_metadata, new_channel_name)}\n{SUMMARY_INSTRUCTIONS}"
                    response_json = insights_client.describe_channel_json(
                        new_channel_name, context=context
                    )
                    structured_description = json.loads(response_json)

                    enrichment_payload = {
                        "channel_id": new_channel_id,
                        "channel_name": new_channel_name,
                        "channel_url": new_channel_url or channel_metadata.get("channel_url"),
                        "channel_description": structured_description.get(
                            "overview", channel_metadata.get("description")
                        ),
                        "channel_description_structured": structured_description,
                        "channel_metadata": channel_metadata,
                        "openai_raw_response": response_json,
                    }

                    metadata_gpt_enrichment = semantic_client.generate_channel_embedding(
                        enrichment_payload
                    )

                except Exception as ai_exc:
                    ai_status = "fallback"
                    log(
                        "⚠️",
                        f"Row {row_num}: AI enrichment failed ({ai_exc}). "
                        "Using parent enrichment as fallback.",
                        report,
                    )
                    metadata_gpt_enrichment = copy.deepcopy(
                        original_data.get("metadata-gpt-enrichment", {})
                    )

                mirrored_data = {
                    "flags": flags,
                    "matching-criteria": copy.deepcopy(original_data.get("matching-criteria", {})),
                    "metadata-gpt-enrichment": metadata_gpt_enrichment,
                    "metadata-keywords": copy.deepcopy(original_data.get("metadata-keywords", {})),
                    "name": new_channel_name,
                    "schedule": copy.deepcopy(original_data.get("schedule", {})),
                    "services": copy.deepcopy(original_data.get("services", {})),
                    "sidekick": {}, # Do not copy sidekick from parent; new channel gets its own
                    "client": original_data.get("client", "NETWORK"),
                    "parent-channel-id": parent_id,
                }

                if dry_run:
                    action = "UPDATE" if target_existed else "CREATE"
                    log("🏜️", f"[DRY-RUN] Would {action}: {new_channel_id} (AI: {ai_status})", report)
                    count += 1
                    continue

                try:
                    target_ref.set(mirrored_data, merge=True)
                    action = "updated" if target_existed else "created"
                    log("🔥", f"Row {row_num}: Firestore {action}: {parent_id} → {new_channel_id}", report)
                except Exception as fs_err:
                    log("❌", f"Row {row_num}: Firestore write failed ({fs_err}) — skipped", report)
                    continue

                tables = [
                    {"name": "channel_embeddings", "id_field": "channel_id", "name_field": "channel_name"},
                    {"name": "channel_video_embeddings", "id_field": "channel_id", "name_field": None},
                ]

                bq_success = True
                try:
                    for t in tables:
                        table_id = f"{bq_project}.{bq_dataset}.{t['name']}"

                        # 1. Compare server-side
                        is_same = bq_rows_are_equal(
                            bq_client=bq_client,
                            table_id=table_id,
                            id_field=t["id_field"],
                            parent_id=parent_id,
                            new_channel_id=new_channel_id,
                            report=report,
                            row_num=row_num,
                            name_field=t["name_field"],
                            new_channel_name=new_channel_name,
                        )

                        if is_same:
                            continue

                        # 2. Sync server-side
                        log(
                            "🔄",
                            f"Row {row_num}: syncing BQ table '{t['name']}' server-side",
                            report,
                        )

                        sync_bq_table_server_side(
                            bq_client=bq_client,
                            table_id=table_id,
                            id_field=t["id_field"],
                            parent_id=parent_id,
                            new_channel_id=new_channel_id,
                            report=report,
                            row_num=row_num,
                            name_field=t["name_field"],
                            new_channel_name=new_channel_name,
                        )

                except Exception as bq_err:
                    bq_success = False
                    log(
                        "❌",
                        f"Row {row_num}: BigQuery failed ({bq_err}) — rolling back Firestore",
                        report,
                    )
                    try:
                        if target_snapshot is not None:
                            target_ref.set(target_snapshot)
                            log("↩️", f"Row {row_num}: Firestore restored to previous state", report)
                        else:
                            target_ref.delete()
                            log("↩️", f"Row {row_num}: Firestore document deleted (was new)", report)
                    except Exception as rb_err:
                        log(
                            "🚨",
                            f"Row {row_num}: ROLLBACK FAILED ({rb_err}) — manual cleanup needed for {new_channel_id}",
                            report,
                        )
                    continue

                status = "✨" if bq_success else "⚠️"
                log(
                    status,
                    f"Row {row_num}: mirrored {parent_id} → {new_channel_id} "
                    f"('{new_channel_name}', AI: {ai_status})",
                    report,
                )
                count += 1

    finally:
        if insights_client:
            insights_client.close()

    print("\n" + "=" * 60)
    print(f"  Mirror complete. Processed: {count} channel(s)")
    if dry_run:
        print("  Mode: DRY RUN — no writes were made")
    print("=" * 60)

    report_path = f"mirror_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved to: {report_path}\n")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mirror YouTube channels from parent to a new channel ID"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be created/updated without writing anything",
    )

    args = parser.parse_args()
    
    # Get mirror.csv directly from the same directory as this script
    csv_path = os.path.join(os.path.dirname(__file__), "mirror.csv")
    mirror_channels(csv_path, dry_run=args.dry_run)
