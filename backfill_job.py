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

SHEET_TAB_INSIGHTS = "Insights_v5"
SHEET_TAB_ACCOUNT_SUMMARY = "Account_Summary"
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
    last_error = None

    for attempt in range(1, 3):
        retry_note = ""
        if attempt > 1:
            retry_note = (
                "\n\nIMPORTANT: Your previous response could not be parsed. "
                "Return ONLY complete, valid JSON. Be concise enough to finish "
                "the entire JSON object. Do not use markdown fences."
            )

        message = anthropic_client.messages.create(
            model=MODEL_EXTRACT,
            max_tokens=8192 if attempt == 1 else 16000,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Here is the full transcript:\n\n{transcript_text}"
                        f"{retry_note}"
                    ),
                }
            ],
        )

        raw = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", "") == "text"
        ).strip()

        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        if message.stop_reason == "max_tokens":
            last_error = RuntimeError(
                f"Claude response truncated at max_tokens on attempt {attempt}"
            )
            print(str(last_error))
            continue

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = e
            print(
                f"JSON parse failed on attempt {attempt}/2: "
                f"{type(e).__name__}: {e}"
            )
            continue

        result["_tokens_used"] = (
            message.usage.input_tokens + message.usage.output_tokens
        )
        result["_source_file"] = source_file
        return result

    raise RuntimeError(
        f"Claude returned invalid JSON after 2 attempts: {last_error}"
    )

def arr_to_str(val):
    if isinstance(val, list):
        # Preserve structured v5 objects as valid JSON in the Sheet.
        if any(isinstance(v, dict) for v in val):
            return json.dumps(val, ensure_ascii=False)
        return " | ".join(str(v) for v in val if v)

    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)

    return val if val is not None else ""


def write_to_sheets(sheets, call_id, file_id, extraction):
    now = datetime.utcnow().isoformat()

    insight_row = [[
        call_id,
        extraction.get("call_type", ""),
        extraction.get("confidence", ""),
        extraction.get("customer_name", ""),
        extraction.get("call_date", ""),
        extraction.get("call_duration_minutes", ""),
        arr_to_str(extraction.get("participants_pivotree", [])),
        arr_to_str(extraction.get("participants_customer", [])),
        arr_to_str(extraction.get("customer_pain_points", [])),
        arr_to_str(extraction.get("customer_requests", [])),
        arr_to_str(extraction.get("buying_signals", [])),
        arr_to_str(extraction.get("expansion_signals", [])),
        arr_to_str(extraction.get("missed_opportunities", [])),
        arr_to_str(extraction.get("sales_objections", [])),
        arr_to_str(extraction.get("proposal_feedback", [])),
        arr_to_str(extraction.get("delivery_risks", [])),
        arr_to_str(extraction.get("churn_signals", [])),
        arr_to_str(extraction.get("pivotree_improvement_opportunities", [])),
        arr_to_str(extraction.get("partner_mentions", [])),
        arr_to_str(extraction.get("competitive_mentions", [])),
        arr_to_str(extraction.get("ai_mentions", [])),
        extraction.get("_source_file", ""),
        now,
        extraction.get("schema_version", "v5"),
        extraction.get("customer_sentiment", "unclear"),
        extraction.get("sentiment_score", ""),
        extraction.get("sentiment_reason", ""),
        extraction.get("sentiment_evidence", ""),
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



def refresh_account_summary(sheets):
    print("Refreshing Account_Summary")

    _, rows = read_sheet(sheets, SHEET_TAB_INSIGHTS)

    valid_sentiments = {"positive", "neutral", "negative", "mixed"}
    accounts = {}

    for idx, row in enumerate(rows):
        customer_name = str(row.get("customer_name", "")).strip()
        sentiment = str(row.get("customer_sentiment", "")).strip().lower()
        raw_score = str(row.get("sentiment_score", "")).strip()

        if not customer_name:
            continue

        if sentiment not in valid_sentiments:
            continue

        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue

        accounts.setdefault(customer_name, []).append({
            "sentiment": sentiment,
            "score": score,
            "call_date": str(row.get("call_date", "")).strip(),
            "processed_at": str(row.get("processed_at", "")).strip(),
            "row_order": idx,
        })

    summary_rows = [[
        "customer_name",
        "current_sentiment",
        "average_sentiment_score",
        "trend_indicator",
    ]]

    for customer_name in sorted(accounts, key=str.lower):
        calls = accounts[customer_name]

        # YYYY-MM-DD and ISO processed_at values sort correctly as strings.
        calls.sort(
            key=lambda c: (
                c["call_date"],
                c["processed_at"],
                c["row_order"],
            ),
            reverse=True,
        )

        recent = calls[:3]

        # Mode of latest 3. If tied, the most recent call wins.
        counts = {}
        for call in recent:
            sent = call["sentiment"]
            counts[sent] = counts.get(sent, 0) + 1

        max_count = max(counts.values())
        tied = {
            sentiment
            for sentiment, count in counts.items()
            if count == max_count
        }

        current_sentiment = next(
            call["sentiment"]
            for call in recent
            if call["sentiment"] in tied
        )

        avg_recent = sum(c["score"] for c in recent) / len(recent)
        avg_recent = round(avg_recent, 2)

        previous = calls[3:6]

        if not previous:
            trend = "Insufficient History"
        else:
            avg_previous = (
                sum(c["score"] for c in previous) / len(previous)
            )

            change = avg_recent - avg_previous

            if change >= 0.25:
                trend = "Improving"
            elif change <= -0.25:
                trend = "Declining"
            else:
                trend = "Stable"

        summary_rows.append([
            customer_name,
            current_sentiment,
            avg_recent,
            trend,
        ])

    # Rebuild the account summary atomically from the source-of-truth rows.
    sheets.spreadsheets().values().clear(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{SHEET_TAB_ACCOUNT_SUMMARY}!A:D",
        body={},
    ).execute()

    sheets.spreadsheets().values().update(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{SHEET_TAB_ACCOUNT_SUMMARY}!A1",
        valueInputOption="RAW",
        body={"values": summary_rows},
    ).execute()

    print(
        f"Account_Summary refreshed: "
        f"{len(summary_rows) - 1} accounts"
    )



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

    try:
        refresh_account_summary(sheets)
    except Exception as e:
        print(f"ACCOUNT SUMMARY ERROR: {str(e)}")

    print(f"Done. Success={success}, Errors={errors}, Skipped={skipped}")


if __name__ == "__main__":
    main()
