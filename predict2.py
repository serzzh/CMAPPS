from model import *
from config import *
from data_processing import *
import pandas as pd
today = datetime.date.today()

config = TsConf()

import psycopg2
conn = psycopg2.connect(dbname='elevator', user='elevator', 
                        password='elevator', host='95.213.193.6', port = '9432')
cursor = conn.cursor()

cursor.execute('SELECT * FROM PUBLIC.PARAMETERS')
records = cursor.fetchall()
...
cursor.close()
conn.close()

df = pd.DataFrame(records)

config.columns_old = ['FILEID', 'ENGINEID', 'TIMECYCLE', 
                      'Alt', 'Mach', 'TRA', 
                      'Total temp at LPC out (T24)', 'Total temp at HPC out (T30)', 
                      'Total temp at LPT out (T50)', 'Physical core speed (Nc)', 
                      'Static pres at HPC out (Ps30)', 'Corrected core speed (NRc)', 
                      'Bypass Ratio (BPR)', 'Bleed Enthalpy (htBleed)', 'RUL', 'CLUSTER']

df.columns = config.columns_old

df = TsDataFrame(df)

print(type(df))
print(df.iloc[1,])

