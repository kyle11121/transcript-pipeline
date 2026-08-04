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
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON")
DRIVE_FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]

MODEL_EXTRACT = "claude-haiku-4-5-20251001"

SHEET_TAB_INSIGHTS = "Insights"
SHEET_TAB_QUOTES = "Quotes"
SHEET_TAB_LOG = "ProcessingLog"

MAX_FILES = int(os.environ.get("MAX_FILES", "0"))
SLEEP_SECONDS = float(os.environ.get("SLEEP_SECONDS", "2"))

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

with open("extraction_prompt.txt", "r") as f:
    EXTRACTION_SYSTEM_PROMPT = f.read()


def get_services():
    scopes = [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/spreadsheets",
    ]

    if GOOGLE_CREDS_JSON:
        creds_dict = json.loads(GOOGLE_CREDS_JSON)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=scopes,
        )
    elif Path("service-account.json").exists():
        creds = service_account.Credentials.from_service_account_file(
            "service-account.json",
            scopes=scopes,
        )
    else:
        import google.auth
        creds, _ = google.auth.default(scopes=scopes)

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

    for row in rows[1:]:
        record = dict(zip(headers, row + [""] * (len(headers) - len(row))))
        data.append(record)

    return headers, data


def list_drive_files(drive):
    files = []
    page_token = None

    query = f"'{DRIVE_FOLDER_ID}' in parents and trashed = false"

    while True:
        response = drive.files().list(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id, name, modifiedTime)",
            pageSize=1000,
            pageToken=page_token,
            orderBy="modifiedTime desc",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")

        if not page_token:
            break

    return files


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


def write_to_sheets(sheets, call_id, file_id, extraction):
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
            file_id,
        ]]},
    ).execute()


def main():
    print("Starting Cloud Run transcript job")

    drive, sheets = get_services()

    _, log_rows = read_sheet(sheets, SHEET_TAB_LOG)

    processed_file_ids = {
        str(r.get("file_id", "")).strip()
        for r in log_rows
        if str(r.get("status", "")).strip().lower() == "success"
        and str(r.get("file_id", "")).strip()
    }

    drive_files = list_drive_files(drive)

    pending = []

    for file in drive_files:
        file_id = file.get("id", "").strip()
        file_name = file.get("name", "").strip()

        if not file_name.lower().endswith(".txt"):
            continue

        if file_id in processed_file_ids:
            continue

        pending.append({
            "file_id": file_id,
            "file_name": file_name,
        })

    if MAX_FILES > 0:
        pending = pending[:MAX_FILES]

    print(f"Drive files found: {len(drive_files)}")
    print(f"Pending files selected: {len(pending)}")

    success = 0
    errors = 0
    skipped = 0

    for row in pending:
        file_id = row["file_id"]
        file_name = row["file_name"]
        call_id = (
            file_name[:-4] if file_name.lower().endswith(".txt") else file_name
        ).strip()

        print(f"Processing: {file_name}")

        try:
            transcript_text = download_file_text(drive, file_id)

            if len(transcript_text.strip()) < 200:
                skipped += 1
                print(f"Skipped: {file_name} — transcript too short")
                continue

            extraction = extract_transcript(transcript_text, file_name)
            write_to_sheets(sheets, call_id, file_id, extraction)

            success += 1
            print(f"Success: {file_name}")

        except Exception as e:
            errors += 1
            print(f"ERROR {file_name}: {str(e)}")

        time.sleep(SLEEP_SECONDS)

    print(f"Done. Success={success}, Errors={errors}, Skipped={skipped}")


if __name__ == "__main__":
    main()
