"""
Transcript Extraction API
--------------------------
Flask API that accepts a Google Drive file ID, downloads the transcript,
runs Claude extraction, and writes results to Google Sheets.

Workato calls POST /process-transcript with:
  { "file_id": "...", "file_name": "..." }

Deploy to Render as a web service.
"""

import os
import io
import json
import re
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify
import anthropic
from google.oauth2 import service_account
from googleapiclient.discovery import build

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY")
GOOGLE_SHEET_ID    = os.environ.get("GOOGLE_SHEET_ID")
GOOGLE_CREDS_JSON  = os.environ.get("GOOGLE_CREDENTIALS_JSON")  # full JSON as env var string
API_SECRET         = os.environ.get("API_SECRET", "")  # optional auth token
MODEL              = "claude-haiku-4-5-20251001"
SHEET_TAB_INSIGHTS = "Insights"
SHEET_TAB_QUOTES   = "Quotes"
SHEET_TAB_LOG      = "ProcessingLog"

# ── Load extraction prompt ────────────────────────────────────────────────────

PROMPT_PATH = Path(__file__).parent / "extraction_prompt.txt"
with open(PROMPT_PATH, "r") as f:
    EXTRACTION_SYSTEM_PROMPT = f.read()

# ── Clients ───────────────────────────────────────────────────────────────────

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def get_drive_service():
    creds_dict = json.loads(GOOGLE_CREDS_JSON)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive.readonly",
                "https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("drive", "v3", credentials=creds), build("sheets", "v4", credentials=creds)


# ── Core functions ────────────────────────────────────────────────────────────

def download_file_text(drive_service, file_id: str) -> str:
    """Download a .txt file from Google Drive and return as string."""
    request_obj = drive_service.files().get_media(fileId=file_id)
    content = request_obj.execute()
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)


def extract_transcript(transcript_text: str, source_file: str) -> dict:
    """Send transcript to Claude, return parsed JSON extraction."""
    message = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Here is the full transcript:\n\n{transcript_text}"
            }
        ]
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


def write_to_sheets(sheets_service, call_id: str, extraction: dict):
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
        arr_to_str(extraction.get("customer_concerns", [])),
        arr_to_str(extraction.get("objections", [])),
        arr_to_str(extraction.get("feature_requests", [])),
        arr_to_str(extraction.get("product_gaps", [])),
        arr_to_str(extraction.get("proposal_feedback", [])),
        arr_to_str(extraction.get("implementation_risks", [])),
        arr_to_str(extraction.get("timeline_risks", [])),
        arr_to_str(extraction.get("resource_constraints", [])),
        arr_to_str(extraction.get("competitor_mentions", [])),
        arr_to_str(extraction.get("buying_signals", [])),
        arr_to_str(extraction.get("positive_signals", [])),
        extraction.get("_source_file", ""),
        now,
        extraction.get("schema_version", "v3"),
    ]]

    sheets_service.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{SHEET_TAB_INSIGHTS}!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": insight_row}
    ).execute()

    quotes = extraction.get("key_quotes", [])
    if quotes:
        quote_rows = [
            [
                call_id,
                extraction.get("customer_name", ""),
                extraction.get("call_date", ""),
                extraction.get("call_type", ""),
                q.get("text", ""),
                q.get("category", ""),
                q.get("speaker_role", ""),
                q.get("timestamp", ""),
            ]
            for q in quotes
        ]
        sheets_service.spreadsheets().values().append(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=f"{SHEET_TAB_QUOTES}!A1",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": quote_rows}
        ).execute()

    log_row = [[
        call_id,
        extraction.get("_source_file", ""),
        "success",
        "",
        extraction.get("_tokens_used", ""),
        now,
    ]]
    sheets_service.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{SHEET_TAB_LOG}!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": log_row}
    ).execute()


def write_error_log(sheets_service, call_id: str, source_file: str, error: str):
    now = datetime.utcnow().isoformat()
    log_row = [[call_id, source_file, "error", error[:500], "", now]]
    sheets_service.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{SHEET_TAB_LOG}!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": log_row}
    ).execute()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/process-transcript", methods=["POST"])
def process_transcript():
    # Optional auth check
    if API_SECRET:
        token = request.headers.get("X-API-Secret", "")
        if token != API_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    file_id = data.get("file_id")
    file_name = data.get("file_name", file_id)

    if not file_id:
        return jsonify({"error": "file_id is required"}), 400

    call_id = Path(file_name).stem

    try:
        drive_service, sheets_service = get_drive_service()

        transcript_text = download_file_text(drive_service, file_id)

        if len(transcript_text.strip()) < 200:
            return jsonify({
                "status": "skipped",
                "call_id": call_id,
                "reason": "transcript too short"
            }), 200

        extraction = extract_transcript(transcript_text, file_name)
        write_to_sheets(sheets_service, call_id, extraction)

        return jsonify({
            "status": "success",
            "call_id": call_id,
            "call_type": extraction.get("call_type"),
            "tokens_used": extraction.get("_tokens_used"),
        }), 200

    except Exception as e:
        try:
            _, sheets_service = get_drive_service()
            write_error_log(sheets_service, call_id, file_name, str(e))
        except Exception:
            pass

        return jsonify({
            "status": "error",
            "call_id": call_id,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
