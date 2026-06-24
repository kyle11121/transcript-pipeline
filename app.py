"""
Transcript Intelligence - Full Service
----------------------------------------
Endpoints:
  GET  /health                  — liveness check
  POST /process-transcript      — Workato calls this (requires X-API-Secret header)
  GET  /                        — Web UI (requires APP_PASSWORD)
  POST /ask                     — Query endpoint for web UI
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for
import anthropic
from google.oauth2 import service_account
from googleapiclient.discovery import build

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-in-production")

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY")
GOOGLE_SHEET_ID    = os.environ.get("GOOGLE_SHEET_ID")
GOOGLE_CREDS_JSON  = os.environ.get("GOOGLE_CREDENTIALS_JSON")
API_SECRET         = os.environ.get("API_SECRET", "")
APP_PASSWORD       = os.environ.get("APP_PASSWORD", "pivotree2026")
DRIVE_FOLDER_ID    = os.environ.get("DRIVE_FOLDER_ID", "")
MODEL_EXTRACT      = "claude-haiku-4-5-20251001"
MODEL_QUERY        = "claude-sonnet-4-6"
SHEET_TAB_INSIGHTS = "Insights"
SHEET_TAB_QUOTES   = "Quotes"
SHEET_TAB_LOG      = "ProcessingLog"

# ── Load extraction prompt ────────────────────────────────────────────────────

PROMPT_PATH = Path(__file__).parent / "extraction_prompt.txt"
with open(PROMPT_PATH, "r") as f:
    EXTRACTION_SYSTEM_PROMPT = f.read()

QUERY_SYSTEM_PROMPT = """You are an analyst for Pivotree, a B2B commerce services company.
You have access to structured extractions from customer and prospect call transcripts.
Each record includes call type, customer name, date, and extracted fields covering
concerns, objections, product gaps, feature requests, risks, and key quotes.

When answering questions:
- Be direct and specific. Lead with the answer.
- Rank or group findings by frequency and severity when relevant.
- Always cite the specific customer and call where evidence comes from.
- Include verbatim quotes when they exist and are relevant.
- Separate findings by call type when the question benefits from it.
- If the data doesn't support a confident answer, say so plainly.
- Format your response with clear headers and bullets. Keep it tight.
- Never fabricate evidence. Only use what's in the records provided."""

# ── Clients ───────────────────────────────────────────────────────────────────

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def get_services():
    creds_dict = json.loads(GOOGLE_CREDS_JSON)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=[
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/spreadsheets"
        ]
    )
    drive  = build("drive",  "v3", credentials=creds)
    sheets = build("sheets", "v4", credentials=creds)
    return drive, sheets


# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def require_api_secret(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if API_SECRET:
            token = request.headers.get("X-API-Secret", "")
            if token != API_SECRET:
                return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ── Core extraction functions ─────────────────────────────────────────────────

def download_file_text(drive_service, file_id: str) -> str:
    request_obj = drive_service.files().get_media(fileId=file_id)
    content = request_obj.execute()
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)


def extract_transcript(transcript_text: str, source_file: str) -> dict:
    message = anthropic_client.messages.create(
        model=MODEL_EXTRACT,
        max_tokens=4096,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Here is the full transcript:\n\n{transcript_text}"}]
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
        sheets_service.spreadsheets().values().append(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=f"{SHEET_TAB_QUOTES}!A1",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": quote_rows}
        ).execute()

    sheets_service.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{SHEET_TAB_LOG}!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": [[call_id, extraction.get("_source_file", ""), "success", "",
                          extraction.get("_tokens_used", ""), now]]}
    ).execute()


def write_error_log(sheets_service, call_id: str, source_file: str, error: str):
    now = datetime.utcnow().isoformat()
    sheets_service.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{SHEET_TAB_LOG}!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": [[call_id, source_file, "error", error[:500], "", now]]}
    ).execute()


# ── Query functions ───────────────────────────────────────────────────────────

def read_sheet(sheets_service, tab):
    result = sheets_service.spreadsheets().values().get(
        spreadsheetId=GOOGLE_SHEET_ID,
        range=f"{tab}!A1:Z10000"
    ).execute()
    rows = result.get("values", [])
    if not rows:
        return []
    headers = rows[0]
    return [dict(zip(headers, row + [""] * (len(headers) - len(row)))) for row in rows[1:]]


def build_context(insights, quotes):
    quote_map = {}
    for q in quotes:
        cid = q.get("call_id", "")
        if cid not in quote_map:
            quote_map[cid] = []
        quote_map[cid].append(q)

    blocks = []
    for row in insights:
        cid = row.get("call_id", "")
        call_quotes = quote_map.get(cid, [])
        quote_lines = ""
        if call_quotes:
            quote_lines = "\n  Key quotes:\n" + "\n".join(
                f'    [{q.get("category","")} | {q.get("speaker_role","")} | {q.get("timestamp","")}] '
                f'"{q.get("quote_text","")}"'
                for q in call_quotes
            )
        blocks.append(f"""---
Call: {cid}
Customer: {row.get("customer_name", "")}
Date: {row.get("call_date", "")}
Type: {row.get("call_type", "")}
Customer concerns: {row.get("customer_concerns", "")}
Objections: {row.get("objections", "")}
Feature requests: {row.get("feature_requests", "")}
Product gaps: {row.get("product_gaps", "")}
Proposal feedback: {row.get("proposal_feedback", "")}
Implementation risks: {row.get("implementation_risks", "")}
Timeline risks: {row.get("timeline_risks", "")}
Resource constraints: {row.get("resource_constraints", "")}
Competitor mentions: {row.get("competitor_mentions", "")}
Buying signals: {row.get("buying_signals", "")}
Positive signals: {row.get("positive_signals", "")}
{quote_lines}""")
    return "\n".join(blocks)


# ── Web UI ────────────────────────────────────────────────────────────────────

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Transcript Intelligence</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f1117; color: #e8e8e8; display: flex;
           align-items: center; justify-content: center; min-height: 100vh; }
    .card { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 12px;
            padding: 40px; width: 360px; }
    h1 { font-size: 20px; font-weight: 600; margin-bottom: 8px; }
    p { color: #888; font-size: 14px; margin-bottom: 28px; }
    input { width: 100%; padding: 12px 16px; background: #0f1117;
            border: 1px solid #2a2d3a; border-radius: 8px; color: #e8e8e8;
            font-size: 15px; margin-bottom: 12px; outline: none; }
    input:focus { border-color: #5b6af0; }
    button { width: 100%; padding: 12px; background: #5b6af0; color: white;
             border: none; border-radius: 8px; font-size: 15px;
             font-weight: 500; cursor: pointer; }
    button:hover { background: #4a59df; }
    .error { color: #ff6b6b; font-size: 13px; margin-bottom: 12px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Transcript Intelligence</h1>
    <p>Pivotree internal tool. Enter password to continue.</p>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form method="POST">
      <input type="password" name="password" placeholder="Password" autofocus>
      <button type="submit">Sign in</button>
    </form>
  </div>
</body>
</html>
"""

APP_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Transcript Intelligence</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f1117; color: #e8e8e8; }
    .header { background: #1a1d27; border-bottom: 1px solid #2a2d3a;
              padding: 16px 32px; display: flex; align-items: center;
              justify-content: space-between; }
    .header h1 { font-size: 17px; font-weight: 600; }
    .header span { color: #888; font-size: 13px; }
    .logout { color: #888; font-size: 13px; text-decoration: none; }
    .logout:hover { color: #e8e8e8; }
    .main { max-width: 860px; margin: 0 auto; padding: 40px 24px; }
    .search-box { display: flex; gap: 12px; margin-bottom: 32px; }
    textarea { flex: 1; padding: 14px 16px; background: #1a1d27;
               border: 1px solid #2a2d3a; border-radius: 10px; color: #e8e8e8;
               font-size: 15px; resize: none; height: 60px; outline: none;
               font-family: inherit; line-height: 1.5; }
    textarea:focus { border-color: #5b6af0; }
    button { padding: 14px 24px; background: #5b6af0; color: white;
             border: none; border-radius: 10px; font-size: 15px;
             font-weight: 500; cursor: pointer; white-space: nowrap; }
    button:hover { background: #4a59df; }
    button:disabled { background: #2a2d3a; color: #555; cursor: not-allowed; }
    .suggestions { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 32px; }
    .chip { padding: 6px 14px; background: #1a1d27; border: 1px solid #2a2d3a;
            border-radius: 20px; font-size: 13px; color: #aaa; cursor: pointer; }
    .chip:hover { border-color: #5b6af0; color: #e8e8e8; }
    .answer { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 10px;
              padding: 28px; line-height: 1.7; font-size: 15px; }
    .answer h2, .answer h3 { margin: 20px 0 8px; font-size: 15px; }
    .answer h2:first-child, .answer h3:first-child { margin-top: 0; }
    .answer ul, .answer ol { padding-left: 20px; margin: 8px 0; }
    .answer li { margin-bottom: 6px; }
    .answer strong { color: #fff; }
    .answer hr { border: none; border-top: 1px solid #2a2d3a; margin: 16px 0; }
    .answer table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 14px; }
    .answer th { text-align: left; padding: 8px 12px; background: #0f1117;
                 border: 1px solid #2a2d3a; color: #aaa; }
    .answer td { padding: 8px 12px; border: 1px solid #2a2d3a; }
    .meta { color: #555; font-size: 12px; margin-top: 16px; }
    .loading { color: #888; font-size: 14px; padding: 20px 0; }
    .spinner { display: inline-block; width: 14px; height: 14px;
               border: 2px solid #333; border-top-color: #5b6af0;
               border-radius: 50%; animation: spin 0.8s linear infinite;
               margin-right: 8px; vertical-align: middle; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .filter { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
    .filter label { font-size: 13px; color: #888; align-self: center; }
    select { padding: 6px 12px; background: #1a1d27; border: 1px solid #2a2d3a;
             border-radius: 6px; color: #e8e8e8; font-size: 13px; outline: none; }
  </style>
</head>
<body>
  <div class="header">
    <h1>Transcript Intelligence</h1>
    <div style="display:flex;gap:20px;align-items:center">
      <span id="record-count">Loading...</span>
      <a href="/logout" class="logout">Sign out</a>
    </div>
  </div>
  <div class="main">
    <div class="suggestions">
      <div class="chip" onclick="setQ('What product gaps are showing up repeatedly?')">Product gaps</div>
      <div class="chip" onclick="setQ('What are the biggest customer complaints?')">Top complaints</div>
      <div class="chip" onclick="setQ('What are the most common objections in sales calls?')">Sales objections</div>
      <div class="chip" onclick="setQ('What are customers asking for that we don\\'t have?')">Feature requests</div>
      <div class="chip" onclick="setQ('What are the biggest implementation risks right now?')">Implementation risks</div>
      <div class="chip" onclick="setQ('What are customers saying about competitors?')">Competitor mentions</div>
      <div class="chip" onclick="setQ('What feedback are customers giving on our proposals or SOWs?')">Proposal feedback</div>
    </div>
    <div class="filter">
      <label>Filter by call type:</label>
      <select id="call-type-filter">
        <option value="">All call types</option>
        <option value="sales_discovery">Sales Discovery</option>
        <option value="sales_followup">Sales Follow-up</option>
        <option value="delivery">Delivery</option>
        <option value="renewal">Renewal</option>
        <option value="escalation">Escalation</option>
        <option value="qbr">QBR</option>
      </select>
    </div>
    <div class="search-box">
      <textarea id="question" placeholder="Ask anything about your customer calls..." rows="2"></textarea>
      <button id="ask-btn" onclick="askQuestion()">Ask</button>
    </div>
    <div id="result"></div>
  </div>

  <script>
    // Load record count on page load
    fetch('/stats').then(r=>r.json()).then(d=>{
      document.getElementById('record-count').textContent = d.total_calls + ' calls';
    }).catch(()=>{
      document.getElementById('record-count').textContent = '';
    });

    function setQ(q) {
      document.getElementById('question').value = q;
      document.getElementById('question').focus();
    }

    document.getElementById('question').addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
      }
    });

    function askQuestion() {
      const q = document.getElementById('question').value.trim();
      if (!q) return;
      const callType = document.getElementById('call-type-filter').value;
      const btn = document.getElementById('ask-btn');
      const result = document.getElementById('result');
      btn.disabled = true;
      btn.textContent = 'Thinking...';
      result.innerHTML = '<div class="loading"><span class="spinner"></span>Reading through your calls...</div>';

      fetch('/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: q, call_type: callType || null})
      })
      .then(r => r.json())
      .then(d => {
        if (d.error) {
          result.innerHTML = '<div class="answer" style="color:#ff6b6b">' + d.error + '</div>';
        } else {
          result.innerHTML = '<div class="answer">' + marked(d.answer) +
            '<div class="meta">' + d.records_searched + ' calls searched</div></div>';
        }
      })
      .catch(e => {
        result.innerHTML = '<div class="answer" style="color:#ff6b6b">Error: ' + e.message + '</div>';
      })
      .finally(() => {
        btn.disabled = false;
        btn.textContent = 'Ask';
      });
    }

    // Simple markdown renderer
    function marked(text) {
      return text
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^---$/gm, '<hr>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/^\* (.+)$/gm, '<li>$1</li>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/(<li>.*<\/li>\n?)+/g, m => '<ul>' + m + '</ul>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');
    }
  </script>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == APP_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("index"))
        error = "Incorrect password."
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template_string(APP_HTML)


@app.route("/stats")
@login_required
def stats():
    try:
        _, sheets = get_services()
        insights = read_sheet(sheets, SHEET_TAB_INSIGHTS)
        return jsonify({"total_calls": len(insights)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
@login_required
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    call_type_filter = data.get("call_type")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        _, sheets = get_services()
        insights = read_sheet(sheets, SHEET_TAB_INSIGHTS)

        if call_type_filter:
            insights = [r for r in insights if r.get("call_type", "").lower() == call_type_filter.lower()]

        if not insights:
            return jsonify({"error": "No records found."}), 200

        call_ids = {r.get("call_id") for r in insights}
        quotes = read_sheet(sheets, SHEET_TAB_QUOTES)
        quotes = [q for q in quotes if q.get("call_id") in call_ids]

        context = build_context(insights, quotes)

        message = anthropic_client.messages.create(
            model=MODEL_QUERY,
            max_tokens=4096,
            system=QUERY_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Here is the extracted intelligence from {len(insights)} calls:\n\n{context}\n\n---\n\nQuestion: {question}"
            }]
        )

        return jsonify({
            "answer": message.content[0].text.strip(),
            "records_searched": len(insights),
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process-transcript", methods=["POST"])
@require_api_secret
def process_transcript():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    file_id = data.get("file_id")
    file_name = data.get("file_name", file_id)

    if not file_id:
        return jsonify({"error": "file_id is required"}), 400

    call_id = Path(file_name).stem

    try:
        drive, sheets = get_services()
        transcript_text = download_file_text(drive, file_id)

        if len(transcript_text.strip()) < 200:
            return jsonify({"status": "skipped", "call_id": call_id, "reason": "too short"}), 200

        extraction = extract_transcript(transcript_text, file_name)
        write_to_sheets(sheets, call_id, extraction)

        return jsonify({
            "status": "success",
            "call_id": call_id,
            "call_type": extraction.get("call_type"),
            "tokens_used": extraction.get("_tokens_used"),
        }), 200

    except Exception as e:
        try:
            _, sheets = get_services()
            write_error_log(sheets, call_id, file_name, str(e))
        except Exception:
            pass
        return jsonify({"status": "error", "call_id": call_id, "error": str(e)}), 500


@app.route("/backfill", methods=["POST"])
@require_api_secret
def backfill():
    """
    Reads all .txt files from the configured Drive folder,
    skips any already in ProcessingLog, processes the rest.
    Call once to bulk-process existing transcripts.
    POST /backfill with X-API-Secret header.
    Optional body: {"limit": 50} to process in batches.
    """
    data = request.get_json() or {}
    limit = int(data.get("limit", 9999))
    folder_id = data.get("folder_id", DRIVE_FOLDER_ID)

    if not folder_id:
        return jsonify({"error": "folder_id required. Set DRIVE_FOLDER_ID env var or pass in body."}), 400

    try:
        drive, sheets = get_services()

        # Get already-processed call IDs from ProcessingLog
        log_rows = read_sheet(sheets, SHEET_TAB_LOG)
        processed_ids = {r.get("call_id", "") for r in log_rows}

        # List all files in the folder
        results = drive.files().list(
            q=f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false",
            fields="files(id, name, mimeType)",
            pageSize=1000
        ).execute()

        all_files = results.get("files", [])

        # Filter to only unprocessed files
        to_process = [
            f for f in all_files
            if Path(f["name"]).stem not in processed_ids
        ][:limit]

        total = len(all_files)
        skipped = total - len(to_process) - (total - len([f for f in all_files if Path(f["name"]).stem not in processed_ids]))
        already_done = total - len([f for f in all_files if Path(f["name"]).stem not in processed_ids])

        results_summary = {
            "total_in_folder": total,
            "already_processed": already_done,
            "to_process": len(to_process),
            "succeeded": 0,
            "failed": 0,
            "errors": []
        }

        for f in to_process:
            file_id = f["id"]
            file_name = f["name"]
            call_id = Path(file_name).stem

            try:
                transcript_text = download_file_text(drive, file_id)

                if len(transcript_text.strip()) < 200:
                    write_error_log(sheets, call_id, file_name, "too short")
                    results_summary["failed"] += 1
                    continue

                extraction = extract_transcript(transcript_text, file_name)
                write_to_sheets(sheets, call_id, extraction)
                results_summary["succeeded"] += 1

            except Exception as e:
                try:
                    write_error_log(sheets, call_id, file_name, str(e))
                except Exception:
                    pass
                results_summary["failed"] += 1
                results_summary["errors"].append({"file": file_name, "error": str(e)[:200]})

        return jsonify(results_summary), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
