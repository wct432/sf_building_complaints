import os
import pandas as pd 
from sodapy import Socrata

working_dir = os.getcwd()
data_path = os.path.dirname(working_dir) + '/data/'

def fetch_data(app_key):
    client = Socrata("data.sfgov.org", app_key)
    results = client.get("gm2e-bten", limit = 300000)
    results_df = pd.DataFrame.from_records(results, index ='complaint_number')
    results_df.to_csv(data_path + 'raw_data.csv')