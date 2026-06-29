import os
import json
import re
import time
from pathlib import Path
from datetime import datetime

import anthropic
from google.oauth2 import service_account
from googleapiclient.discovery import build


ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GOOGLE_SHEET_ID = os.environ["GOOGLE_SHEET_ID"]
GOOGLE_CREDS_JSON = os.environ["GOOGLE_CREDENTIALS_JSON"]

MODEL_EXTRACT = "claude-haiku-4-5-20251001"

SHEET_TAB_INSIGHTS = "Insights"
SHEET_TAB_QUOTES = "Quotes"
SHEET_TAB_LOG = "ProcessingLog"
SHEET_TAB_QUEUE = "FileQueue"

MAX_FILES = int(os.environ.get("MAX_FILES", "25"))  # 0 = all pending
SLEEP_SECONDS = float(os.environ.get("SLEEP_SECONDS", "2"))

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

with open("extraction_prompt.txt", "r") as f:
    EXTRACTION_SYSTEM_PROMPT = f.read()


def get_services():
    creds_dict = json.loads(GOOGLE_CREDS_JSON)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=[
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/spreadsheets",
        ],
    )
    drive = build("drive", "v3", credentials=creds)
    sheets = build("sheets", "v4", credentials=creds)
    return drive, sheets


def read_sheet(sheets, tab):
    result = sheets.spreadsheets().values().get(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{tab}!A1:Z10000",
    ).execute()
    rows = result.get("values", [])
    if not rows:
        return [], []
    headers = rows[0]
    data = []
    for i, row in enumerate(rows[1:], start=2):
        record = dict(zip(headers, row + [""] * (len(headers) - len(row))))
        record["_row_number"] = i
        data.append(record)
    return headers, data


def update_queue_row(sheets, row_number, status, error_message="", tokens_used=""):
    now = datetime.utcnow().isoformat()
    values = [[status, now, error_message[:500], tokens_used]]
    sheets.spreadsheets().values().update(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{SHEET_TAB_QUEUE}!C{row_number}:F{row_number}",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()


def download_file_text(drive, file_id):
    request_obj = drive.files().get_media(fileId=file_id)
    content = request_obj.execute()
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)


def extract_transcript(transcript_text, source_file):
    message = anthropic_client.messages.create(
        model=MODEL_EXTRACT,
        max_tokens=4096,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Here is the full transcript:\n\n{transcript_text}",
            }
        ],
    )

    raw = message.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)
    result["_tokens_used"] = message.usage.input_tokens + message.usage.output_tokens
    result["_source_file"] = source_file
    return result


def arr_to_str(val):
    if isinstance(val, list):
        return " | ".join(str(v) for v in val if v)
    return val if val is not None else ""


def write_to_sheets(sheets, call_id, extraction):
    now = datetime.utcnow().isoformat()

    insight_row = [[
        call_id,
        extraction.get("call_type", ""),
        extraction.get("confidence", ""),
        extraction.get("transcript_quality", ""),
        extraction.get("customer_name", ""),
        extraction.get("call_date", ""),
        extraction.get("call_duration_minutes", ""),
        arr_to_str(extraction.get("participants_pivotree", [])),
        arr_to_str(extraction.get("participants_customer", [])),
        arr_to_str(extraction.get("sales_objections", [])),
        arr_to_str(extraction.get("service_gaps", [])),
        arr_to_str(extraction.get("proposal_feedback", [])),
        arr_to_str(extraction.get("delivery_risks", [])),
        arr_to_str(extraction.get("churn_signals", [])),
        arr_to_str(extraction.get("competitor_mentions", [])),
        extraction.get("_source_file", ""),
        now,
        extraction.get("schema_version", "v4"),
    ]]

    sheets.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{SHEET_TAB_INSIGHTS}!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": insight_row},
    ).execute()

    quotes = extraction.get("key_quotes", [])
    if quotes:
        quote_rows = [[
            call_id,
            extraction.get("customer_name", ""),
            extraction.get("call_date", ""),
            extraction.get("call_type", ""),
            q.get("text", ""),
            q.get("category", ""),
            q.get("speaker_role", ""),
            q.get("timestamp", ""),
        ] for q in quotes]

        sheets.spreadsheets().values().append(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=f"{SHEET_TAB_QUOTES}!A1",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": quote_rows},
        ).execute()

    sheets.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{SHEET_TAB_LOG}!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": [[
            call_id,
            extraction.get("_source_file", ""),
            "success",
            "",
            extraction.get("_tokens_used", ""),
            now,
        ]]},
    ).execute()


def main():
    print("Starting Cloud Run backfill job")

    drive, sheets = get_services()
    _, queue_rows = read_sheet(sheets, SHEET_TAB_QUEUE)

    pending = [
        r for r in queue_rows
        if r.get("status", "").strip().lower() in ("pending", "")
    ]

    if MAX_FILES > 0:
        pending = pending[:MAX_FILES]

    print(f"Pending files selected: {len(pending)}")

    success = 0
    errors = 0
    skipped = 0

    for row in pending:
        file_id = row.get("file_id", "").strip()
        file_name = row.get("file_name", "").strip()
        row_number = row["_row_number"]
        call_id = Path(file_name).stem

        if not file_id:
            update_queue_row(sheets, row_number, "error", "Missing file_id")
            errors += 1
            continue

        print(f"Processing row {row_number}: {file_name}")

        try:
            transcript_text = download_file_text(drive, file_id)

            if len(transcript_text.strip()) < 200:
                update_queue_row(sheets, row_number, "skipped", "Transcript too short")
                skipped += 1
                continue

            extraction = extract_transcript(transcript_text, file_name)
            write_to_sheets(sheets, call_id, extraction)

            update_queue_row(
                sheets,
                row_number,
                "success",
                "",
                str(extraction.get("_tokens_used", "")),
            )

            success += 1
            print(f"Success: {file_name}")

        except Exception as e:
            error_text = str(e)
            print(f"ERROR {file_name}: {error_text}")
            update_queue_row(sheets, row_number, "error", error_text)
            errors += 1

        time.sleep(SLEEP_SECONDS)

    print(f"Done. Success={success}, Errors={errors}, Skipped={skipped}")


if __name__ == "__main__":
    main()
