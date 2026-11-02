
import json

import boto3
from botocore.config import Config

client = boto3.client(
    "lambda",
    region_name="ap-southeast-2",
    config=Config(
        connect_timeout=10,
        read_timeout=700,
    ),
)

response = client.invoke(
    FunctionName="finjk-fundamental-analysis",
    InvocationType="RequestResponse",
    Payload=json.dumps({"ticker": "AAPL"}).encode("utf-8"),
)

payload = json.loads(response["Payload"].read())

print("AWS invocation status:", response["StatusCode"])
print("Lambda function error:", response.get("FunctionError"))
print("Application status:", payload.get("statusCode"))

if payload.get("statusCode") == 200:
    report = json.loads(payload["body"])
    print(json.dumps(report, indent=2))
else:
    print(json.dumps(payload, indent=2))
