import boto3
import pandas as pd 
from sodapy import Socrata

AWS_S3_BUCKET = "sf-building-complaints"
s3_client = boto3.client("s3")   

def fetch_data(app_key):
    #fetch data from the data.sfgov.org Socrata API
    client = Socrata("data.sfgov.org", app_key)
    results = client.get("gm2e-bten", limit = 300000)
    raw_data = pd.DataFrame.from_records(results, index ='complaint_number')
    #save data into our sf-building-complaints bucket
    key = "data/raw_data.csv"
    raw_data.to_csv(
        f"s3://{AWS_S3_BUCKET}/{key}",
        index=False)