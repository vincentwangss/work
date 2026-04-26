"""Test neodata ft_mins API with auth token"""
import urllib.request
import json

TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJteWZFenA3ODNLaV9KQ3g4Vm5jM1hfaXg2alpyYjZDZjVPTWtHWk1QSTNzIn0.eyJleHAiOjE4MDQ2ODMxMTcsImlhdCI6MTc3NzEwOTgxNiwiYXV0aF90aW1lIjoxNzczMTQ3MTE3LCJqdGkiOiIxNzY0ODkyMy1iNjIwLTRiM3EtOTk1MS0xY2Y3MDdkMTBkMGIiLCJpc3MiOiJodHRwczovL3d3dy5jb2RlYnVkZHkuY24vYXV0aC9yZWFsbXMvY29waWxvdCIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiJhOWI3NWJkZi1hZDIwLTQyMTctYjI3NS04OTQyYTcxNDYyOWYiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJjb25zb2xlIiwic2lkIjoiZjcxODA5YmItYjE5NS00NmYxLWE4YzgtYzQ5NWM4YTIwODA2IiwiYWNyIjoiMCIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzIiwib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgb2ZmbGluZV9hY2Nlc3MgZW1haWwiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsIm5pY2tuYW1lIjoi546L5qCR5qOuIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiMTg2MjAyOTYzMTYifQ.kIlgyxZvw20HOFcUOe9osTuIdyhJvWy-EU3OQ2NSa_zm2099cpj1x9vSWhwDmDFCEYRWw7OK4_ugMoFLjYahq3kQXiQHOPYb50w6EI4jpMOD0fgvU9gpilgviUySXiH8rH8yCFu9cpRRCfL-MBSSboJrtT-6864xUI0u89QIdrIgaLep9j3TZkBMeZBZtBG-cSmqZTkp5eu1Vuz0ucZ423g115VpQTfdR9BIfRG0W6aliCb2oaNKv0RbbFlBjoj5WH_8U8V7Ty5dHo0sLW6kXaOgyQrm98Rfzfpm68LWLqlpCWfGuJRyjGdmb-2HxpYJNAa5S6O-BhAyDDNxPZpARg"

url = "https://www.codebuddy.cn/v2/tool/financedata"
payload = {
    "api_name": "ft_mins",
    "params": {
        "ts_code": "IF2606.CFE",
        "freq": "5min",
        "start_date": "2026-03-01 09:00:00",
        "end_date": "2026-04-24 15:00:00",
    },
    "fields": "",
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}",
    },
)

with urllib.request.urlopen(req, timeout=30) as resp:
    raw = resp.read().decode()
    data = json.loads(raw)

print(f"Code: {data.get('code')}")
print(f"Msg: {data.get('msg')}")
d = data.get("data")
if d:
    fields = d.get("fields", [])
    items = d.get("items", [])
    print(f"Fields: {fields}")
    print(f"Total items: {len(items)}")
    if items:
        print(f"First: {items[0]}")
        print(f"Last: {items[-1]}")
else:
    print(f"No data. Response keys: {list(data.keys())}")
