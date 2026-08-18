import base64
import hashlib
import json
import os
import secrets
from urllib.parse import urlencode

import requests
from flask import Blueprint, jsonify, redirect, request, session
from google.cloud import secretmanager
from google.oauth2 import service_account

salesforce_bp = Blueprint("salesforce", __name__)

SALESFORCE_CLIENT_ID = os.environ.get("SALESFORCE_CLIENT_ID", "")
SALESFORCE_MCP_URL = os.environ.get(
    "SALESFORCE_MCP_URL",
    "https://api.salesforce.com/platform/mcp/v1/custom/SalesforceClaudeMCP",
)
SALESFORCE_REDIRECT_URI = os.environ.get(
    "SALESFORCE_REDIRECT_URI",
    "https://transcript-pipeline.onrender.com/salesforce/callback",
)
SALESFORCE_AUTH_URL = os.environ.get(
    "SALESFORCE_AUTH_URL",
    "https://login.salesforce.com/services/oauth2/authorize",
)
SALESFORCE_TOKEN_URL = os.environ.get(
    "SALESFORCE_TOKEN_URL",
    "https://login.salesforce.com/services/oauth2/token",
)
SALESFORCE_REFRESH_SECRET = os.environ.get(
    "SALESFORCE_REFRESH_SECRET",
    "SALESFORCE_REFRESH_TOKEN",
)
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "transcript-pipeline-500417")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")


def _secret_client():
    if not GOOGLE_CREDS_JSON:
        raise RuntimeError("GOOGLE_CREDENTIALS_JSON is not configured")

    creds_info = json.loads(GOOGLE_CREDS_JSON)
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return secretmanager.SecretManagerServiceClient(credentials=creds)


def _secret_parent():
    return f"projects/{GCP_PROJECT_ID}/secrets/{SALESFORCE_REFRESH_SECRET}"


def _read_refresh_token():
    response = _secret_client().access_secret_version(
        request={"name": f"{_secret_parent()}/versions/latest"}
    )
    return response.payload.data.decode("utf-8").strip()


def _store_refresh_token(refresh_token):
    if not refresh_token:
        raise RuntimeError("Salesforce did not return a refresh token")

    _secret_client().add_secret_version(
        request={
            "parent": _secret_parent(),
            "payload": {"data": refresh_token.encode("utf-8")},
        }
    )


def get_salesforce_access_token():
    if not SALESFORCE_CLIENT_ID:
        raise RuntimeError("SALESFORCE_CLIENT_ID is not configured")

    refresh_token = _read_refresh_token()

    response = requests.post(
        SALESFORCE_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": SALESFORCE_CLIENT_ID,
            "refresh_token": refresh_token,
        },
        timeout=20,
    )

    if not response.ok:
        raise RuntimeError(
            f"Salesforce token refresh failed ({response.status_code})"
        )

    token_data = response.json()
    rotated_refresh_token = token_data.get("refresh_token")

    if rotated_refresh_token and rotated_refresh_token != refresh_token:
        _store_refresh_token(rotated_refresh_token)

    access_token = token_data.get("access_token")
    if not access_token:
        raise RuntimeError("Salesforce did not return an access token")

    return access_token


@salesforce_bp.route("/salesforce/connect", methods=["GET"])
def salesforce_connect():
    if not session.get("logged_in"):
        return redirect("/login")

    if not SALESFORCE_CLIENT_ID:
        return jsonify({"error": "SALESFORCE_CLIENT_ID is not configured"}), 500

    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("utf-8")).digest()
    ).rstrip(b"=").decode("ascii")
    state = secrets.token_urlsafe(32)

    session["salesforce_oauth_state"] = state
    session["salesforce_code_verifier"] = verifier

    params = {
        "response_type": "code",
        "client_id": SALESFORCE_CLIENT_ID,
        "redirect_uri": SALESFORCE_REDIRECT_URI,
        "scope": "mcp_api refresh_token",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }

    return redirect(f"{SALESFORCE_AUTH_URL}?{urlencode(params)}")


@salesforce_bp.route("/salesforce/callback", methods=["GET"])
def salesforce_callback():
    if not session.get("logged_in"):
        return redirect("/login")

    if request.args.get("error"):
        return jsonify({
            "error": request.args.get("error"),
            "description": request.args.get("error_description", ""),
        }), 400

    expected_state = session.pop("salesforce_oauth_state", "")
    returned_state = request.args.get("state", "")
    verifier = session.pop("salesforce_code_verifier", "")
    code = request.args.get("code", "")

    if (
        not expected_state
        or not returned_state
        or not secrets.compare_digest(expected_state, returned_state)
    ):
        return jsonify({"error": "Invalid Salesforce OAuth state"}), 400

    if not verifier or not code:
        return jsonify({"error": "Missing OAuth code or PKCE verifier"}), 400

    response = requests.post(
        SALESFORCE_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": SALESFORCE_CLIENT_ID,
            "redirect_uri": SALESFORCE_REDIRECT_URI,
            "code_verifier": verifier,
        },
        timeout=20,
    )

    if not response.ok:
        return jsonify({
            "error": "Salesforce token exchange failed",
            "status": response.status_code,
        }), 400

    token_data = response.json()
    refresh_token = token_data.get("refresh_token")

    if not refresh_token:
        return jsonify({
            "error": "Salesforce authorization succeeded but no refresh token was returned"
        }), 400

    _store_refresh_token(refresh_token)

    return (
        "<h2>Salesforce connected successfully.</h2>"
        "<p>The transcript intelligence app can now refresh Salesforce access "
        "without asking you to approve every lookup.</p>"
        '<p><a href="/">Return to Transcript Intelligence</a></p>'
    )


@salesforce_bp.route("/salesforce/status", methods=["GET"])
def salesforce_status():
    if not session.get("logged_in"):
        return redirect("/login")

    try:
        get_salesforce_access_token()
        return jsonify({
            "connected": True,
            "mcp_url": SALESFORCE_MCP_URL,
        }), 200
    except Exception as exc:
        return jsonify({
            "connected": False,
            "error": str(exc),
        }), 200
