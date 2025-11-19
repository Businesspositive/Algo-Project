from fyers_apiv3 import fyersModel
import webbrowser
import os
import dotenv

dotenv.load_dotenv()

client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")
redirect_uri = os.getenv("redirect_uri")

appSession = fyersModel.SessionModel(client_id = client_id, redirect_uri = redirect_uri,response_type="code",state="default",secret_key=client_secret,grant_type="authorization_code")

auth_code = appSession.generate_authcode()
print(auth_code)