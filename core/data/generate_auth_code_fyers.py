import os

from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

load_dotenv()


def generate_auth_url() -> str:
    client_id = os.getenv("client_id")
    client_secret = os.getenv("client_secret")

    redirect_uri = os.getenv(
        "FYERS_REDIRECT_URI",
        "http://127.0.0.1:5000/webhook",
    )

    if not client_id or not client_secret:
        raise ValueError(
            "client_id and client_secret must be configured in environment variables."
        )

    session = fyersModel.SessionModel(
        client_id=client_id,
        redirect_uri=redirect_uri,
        response_type="code",
        state="default",
        secret_key=client_secret,
        grant_type="authorization_code",
    )

    return session.generate_authcode()


if __name__ == "__main__":
    print("Open the following URL to authorize your Fyers application:")
    print(generate_auth_url())