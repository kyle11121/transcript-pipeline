import json
import os
import re
import time
from datetime import datetime

import anthropic
import google.auth
from googleapiclient.discovery import build

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GOOGLE_SHEET_ID = os.environ["GOOGLE_SHEET_ID"]

MODEL = os.environ.get("SENTIMENT_MODEL", "claude-haiku-4-5-20251001")
SHEET = os.environ.get("SENTIMENT_SHEET", "Insights_v5")
QUOTES_SHEET = os.environ.get("SENTIMENT_QUOTES_SHEET", "Quotes")
BATCH_SIZE = int(os.environ.get("SENTIMENT_BATCH_SIZE", "20"))
SLEEP_SECONDS = float(os.environ.get("SENTIMENT_SLEEP_SECONDS", "1.5"))
MAX_BATCHES = int(os.environ.get("SENTIMENT_MAX_BATCHES", "0"))
MAX_RETRIES = int(os.environ.get("SENTIMENT_MAX_RETRIES", "3"))

SENTIMENT_FIELDS = ["customer_sentiment","sentiment_score","sentiment_reason","sentiment_evidence"]
SOURCE_FIELDS = ["customer_pain_points","customer_requests","buying_signals","expansion_signals","sales_objections","proposal_feedback","delivery_risks","churn_signals","pivotree_improvement_opportunities"]

SYSTEM_PROMPT = """You are classifying historical customer sentiment for Pivotree from EXISTING structured call intelligence. You are NOT reading the original transcript.

Measure only the customer's expressed sentiment TOWARD PIVOTREE, Pivotree people, Pivotree delivery, responsiveness, recommendations, services, or the relationship.

Return ONLY a valid JSON array, one object for every supplied row, in the same order.

Each object must be:
{
  "row_number": 123,
  "customer_sentiment": "positive|neutral|negative|mixed|unclear",
  "sentiment_score": -2,
  "sentiment_reason": "one concise sentence",
  "sentiment_evidence": "short exact customer quote or empty string"
}

Scoring:
2 = strongly positive toward Pivotree
1 = positive toward Pivotree
0 = neutral or mixed
-1 = negative toward Pivotree
-2 = strongly negative toward Pivotree
Use an empty string for sentiment_score only when customer_sentiment is unclear.

Rules:
- Technical problems, delivery complexity, or frustration with a third-party platform are NOT automatically negative sentiment toward Pivotree.
- A normal customer request is not automatically negative sentiment.
- A renewal or buying discussion is not automatically positive sentiment.
- Customer pain points may be unrelated to Pivotree; attribute carefully.
- Use customer statements, not Pivotree employees' interpretation of what the customer thinks.
- Do not convert churn risk, a technical dependency, or an unresolved issue into sentiment unless the evidence supports customer attitude toward Pivotree.
- If there are meaningful positive and negative expressions toward Pivotree in the same call, use mixed and score 0.
- Use strong scores (+2/-2) only for unusually explicit praise/trust or severe dissatisfaction/loss of trust.
- sentiment_evidence may ONLY quote text supplied in customer_quotes. Never invent or reconstruct a quote.
- If there is no reliable supplied customer quote, sentiment_evidence must be an empty string.
- Because this is a historical backfill from structured data rather than the full transcript, use unclear when the available evidence is genuinely insufficient or contradictory.
- Do not infer speaker identity from room/device labels.
"""

def get_sheets():
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("sheets", "v4", credentials=creds)

def read_tab(sheets, tab, range_end):
    result = sheets.spreadsheets().values().get(spreadsheetId=GOOGLE_SHEET_ID, range=f"{tab}!A1:{range_end}").execute()
    rows = result.get("values", [])
    if not rows:
        return [], []
    headers = rows[0]
    data = []
    for idx, row in enumerate(rows[1:], start=2):
        padded = row + [""] * max(0, len(headers) - len(row))
        record = dict(zip(headers, padded))
        record["_row_number"] = idx
        data.append(record)
    return headers, data

def read_quotes(sheets):
    headers, rows = read_tab(sheets, QUOTES_SHEET, "H50000")
    if not headers:
        return {}
    quote_map = {}
    for row in rows:
        role = str(row.get("speaker_role", "")).strip().lower()
        if role not in {"client","customer","prospect"}:
            continue
        call_id = str(row.get("call_id", "")).strip()
        text = str(row.get("text", "") or row.get("quote_text", "")).strip()
        if not call_id or not text:
            continue
        quote_map.setdefault(call_id, [])
        if len(quote_map[call_id]) < 4:
            quote_map[call_id].append(text[:900])
    return quote_map

def compact_text(value, limit=1400):
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"

def build_record(row, quote_map):
    record = {
        "row_number": row["_row_number"],
        "call_id": compact_text(row.get("call_id"), 500),
        "customer_name": compact_text(row.get("customer_name"), 250),
        "call_date": compact_text(row.get("call_date"), 50),
        "call_type": compact_text(row.get("call_type"), 80),
        "customer_quotes": quote_map.get(str(row.get("call_id", "")).strip(), []),
    }
    for field in SOURCE_FIELDS:
        value = compact_text(row.get(field), 1400)
        if value:
            record[field] = value
    return record

def parse_json(text):
    raw = text.strip()
    start_idx = raw.find("[")
    end_idx = raw.rfind("]")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError("Claude response did not contain a JSON array")
    value = json.loads(raw[start_idx:end_idx + 1])
    if not isinstance(value, list):
        raise ValueError("Claude response was not a JSON array")
    return value

def classify_batch(client, records):
    prompt = "Classify the following rows. Return one result for every row_number.\n\n" + json.dumps(records, ensure_ascii=False)
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            msg = client.messages.create(model=MODEL, max_tokens=6000, system=SYSTEM_PROMPT, messages=[{"role":"user","content":prompt}])
            raw_text = "".join(
                block.text for block in msg.content
                if getattr(block, "type", "") == "text"
            )
            try:
                return parse_json(raw_text)
            except Exception:
                print(f"RAW CLAUDE RESPONSE: {raw_text[:3000]!r}")
                print(f"CLAUDE STOP REASON: {msg.stop_reason}")
                raise
        except anthropic.BadRequestError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                raise
            wait = 3 * attempt
            print(f"Transient Claude error; retry {attempt}/{MAX_RETRIES} in {wait}s: {exc}")
            time.sleep(wait)
    raise last_error

def normalize_result(item, allowed_rows):
    row_number = int(item.get("row_number"))
    if row_number not in allowed_rows:
        raise ValueError(f"Unexpected row_number {row_number}")
    sentiment = str(item.get("customer_sentiment", "unclear")).strip().lower()
    if sentiment not in {"positive","neutral","negative","mixed","unclear"}:
        sentiment = "unclear"
    score = item.get("sentiment_score", "")
    if sentiment == "unclear":
        score = ""
    else:
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0
        score = max(-2, min(2, score))
        if sentiment == "positive" and score <= 0:
            score = 1
        elif sentiment == "negative" and score >= 0:
            score = -1
        elif sentiment in {"neutral","mixed"}:
            score = 0
    return {
        "row_number": row_number,
        "customer_sentiment": sentiment,
        "sentiment_score": score,
        "sentiment_reason": compact_text(item.get("sentiment_reason"), 1200),
        "sentiment_evidence": compact_text(item.get("sentiment_evidence"), 1200),
    }

def write_results(sheets, results):
    data = []
    for item in results:
        row = item["row_number"]
        data.append({"range":f"{SHEET}!Y{row}:AB{row}","values":[[item["customer_sentiment"],item["sentiment_score"],item["sentiment_reason"],item["sentiment_evidence"]]]})
    sheets.spreadsheets().values().batchUpdate(spreadsheetId=GOOGLE_SHEET_ID, body={"valueInputOption":"RAW","data":data}).execute()

def main():
    print(f"Starting sentiment backfill at {datetime.utcnow().isoformat()}")
    print(f"Model={MODEL} BatchSize={BATCH_SIZE}")
    sheets = get_sheets()
    headers, rows = read_tab(sheets, SHEET, "AB10000")
    missing = [f for f in SENTIMENT_FIELDS if f not in headers]
    if missing:
        raise RuntimeError(f"Missing sentiment columns in {SHEET}: {missing}")
    quote_map = read_quotes(sheets)
    pending = []
    skipped_internal = 0
    skipped_no_customer = 0
    for row in rows:
        if str(row.get("customer_sentiment", "")).strip():
            continue
        call_type = str(row.get("call_type", "")).strip().lower()
        customer = str(row.get("customer_name", "")).strip()
        if call_type == "internal":
            skipped_internal += 1
            continue
        if not customer or customer.lower() == "pivotree":
            skipped_no_customer += 1
            continue
        pending.append(row)
    print(f"Pending={len(pending)} SkippedInternal={skipped_internal} SkippedNoCustomer={skipped_no_customer}")
    if not pending:
        print("Nothing to backfill.")
        return
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    completed = 0
    batches = 0
    for start in range(0, len(pending), BATCH_SIZE):
        if MAX_BATCHES > 0 and batches >= MAX_BATCHES:
            print(f"Stopped at SENTIMENT_MAX_BATCHES={MAX_BATCHES}")
            break
        batch_rows = pending[start:start + BATCH_SIZE]
        records = [build_record(row, quote_map) for row in batch_rows]
        allowed_rows = {row["_row_number"] for row in batch_rows}
        print(f"Batch {batches + 1}: rows {min(allowed_rows)}-{max(allowed_rows)} ({len(records)} calls)")
        try:
            raw_results = classify_batch(client, records)
            normalized = [normalize_result(item, allowed_rows) for item in raw_results]
            returned_rows = {item["row_number"] for item in normalized}
            if returned_rows != allowed_rows:
                missing_rows = sorted(allowed_rows - returned_rows)
                extra_rows = sorted(returned_rows - allowed_rows)
                raise ValueError(f"Batch row mismatch. Missing={missing_rows} Extra={extra_rows}")
            write_results(sheets, normalized)
            completed += len(normalized)
            batches += 1
            print(f"Saved {len(normalized)}. Total completed this run={completed}")
        except anthropic.BadRequestError as exc:
            print(f"STOPPING on Claude 400: {exc}")
            print("Completed rows remain saved. Rerun after the API limit is resolved; it resumes from blanks.")
            return
        except Exception as exc:
            print(f"ERROR in batch: {type(exc).__name__}: {exc}")
            print("Stopping to avoid skipping rows. Completed rows remain saved; rerun to resume from blanks.")
            return
        time.sleep(SLEEP_SECONDS)
    print(f"Sentiment backfill finished. Completed this run={completed}, Batches={batches}")

if __name__ == "__main__":
    main()
