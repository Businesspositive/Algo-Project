from __future__ import annotations

import os
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd
import requests
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

from utils.database import save_tokens, get_latest_tokens
from .base import DataSource

load_dotenv()


def _session_kwargs() -> Dict[str, Any]:
    redirect_uri = os.getenv("FYERS_REDIRECT_URI", "http://127.0.0.1:5000/webhook")
    client_id = os.getenv("client_id")
    secret_key = os.getenv("client_secret")
    if not client_id or not secret_key:
        raise ValueError("client_id/client_secret must be provided in environment variables.")
    return {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "state": "default",
        "secret_key": secret_key,
        "grant_type": "authorization_code",
    }


def _create_session() -> fyersModel.SessionModel:
    return fyersModel.SessionModel(**_session_kwargs())


def generate_access_token(auth_code: Optional[str] = None) -> str:
    """
    Generate a fresh access token using an auth code (one-time login).
    """
    code = auth_code or os.getenv("FYERS_AUTH_CODE")
    if not code:
        raise ValueError("FYERS_AUTH_CODE env variable must be set for initial token generation.")

    session = _create_session()
    session.set_token(code)
    resp = session.generate_token()
    if resp.get("s") != "ok":
        raise RuntimeError(f"Failed to generate FYERS token: {resp}")
    access_token = resp["access_token"]
    refresh_token = resp.get("refresh_token")
    if refresh_token:
        save_tokens(access_token, refresh_token)
    return access_token


def refresh_access_token(pin: Optional[str] = "5526") -> str:
    """
    Refresh access token using the latest stored refresh token.
    """
    client_id = os.getenv("client_id")
    secret_key = os.getenv("client_secret")
    if not client_id or not secret_key:
        raise ValueError("client_id/client_secret env variables are required.")

    _, refresh_token = get_latest_tokens()
    if not refresh_token:
        raise ValueError("No refresh token found. Run generate_access_token once.")

    concatenated = f"{client_id}:{secret_key}"
    app_id_hash = hashlib.sha256(concatenated.encode()).hexdigest()
    payload = {
        "grant_type": "refresh_token",
        "appIdHash": app_id_hash,
        "refresh_token": refresh_token,
    }
    print(pin, "pin")
    if pin:
        payload["pin"] = pin
    
    print(payload, "payload")

    resp = requests.post(
        "https://api-t1.fyers.in/api/v3/validate-refresh-token",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    print(resp.text)
    resp.raise_for_status()
    body = resp.json()
    if body.get("s") != "ok" or "access_token" not in body:
        raise RuntimeError(f"Refresh token call failed: {body}")
    new_access_token = body["access_token"]
    save_tokens(new_access_token, refresh_token)
    return new_access_token


def get_fyers_client(pin: str = "5526") -> fyersModel.FyersModel:
    """
    Ensure we have a valid authenticated Fyers client.
    """
    client_id = os.getenv("client_id")
    if not client_id:
        raise ValueError("client_id env variable is required to instantiate FyersModel.")

    access_token, _ = get_latest_tokens()
    if not access_token:
        access_token = generate_access_token()

    def _instantiate(token: str):
        return fyersModel.FyersModel(client_id=client_id, token=token, log_path="")

    client = _instantiate(access_token)
    try:
        client.profile()
    except Exception:
        refreshed = refresh_access_token(pin=pin)
        client = _instantiate(refreshed)
    return client


class FyersDataSource(DataSource):
    """
    DataSource implementation backed by Fyers history API.
    Automatically chunks requests to comply with API limits.
    """

    def __init__(self, fyers_client: Optional[fyersModel.FyersModel] = None):
        self._client = fyers_client or get_fyers_client()

    def fetch(
        self,
        symbol: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "D",
    ) -> pd.DataFrame:
        end_date = end or datetime.today().strftime("%Y-%m-%d")
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        frames: List[pd.DataFrame] = []
        current_start = start_dt

        while current_start <= end_dt:
            current_end = min(current_start + timedelta(days=364), end_dt)
            payload = {
                "symbol": self._format_symbol(symbol),
                "resolution": interval,
                "date_format": "1",
                "range_from": current_start.strftime("%Y-%m-%d"),
                "range_to": current_end.strftime("%Y-%m-%d"),
                "cont_flag": "1",
            }
            response = self._client.history(payload)
            candles = response.get("candles", [])
            if not candles:
                break
            df = pd.DataFrame(
                candles,
                columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
            )
            df["Date"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("Date", inplace=True)
            df.drop(columns=["timestamp"], inplace=True)
            frames.append(df)
            current_start = current_end + timedelta(days=1)

        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames).sort_index()
        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return data

    @staticmethod
    def _format_symbol(ticker: str) -> str:
        if ticker.startswith("NSE:"):
            return ticker
        if ticker.endswith(".NS"):
            return f"NSE:{ticker.replace('.NS', '')}-EQ"
        return ticker


