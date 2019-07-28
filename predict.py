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

train = df[(["FILEID","ENGINEID", "TIMECYCLE"]+config.select_feat+["RUL"])][df.FILEID.isin(config.train_fids)]
test = df[(["FILEID","ENGINEID", "TIMECYCLE"]+config.select_feat+["RUL"])][df.FILEID.isin(config.test_fids)]

train.tail()

n_eng_train = max(train.ENGINEID)
n_eng_test = max(test.ENGINEID)

train.target = config.target
train.groupids = config.groupids
train.timestamp = config.timestamp
test.target = config.target
test.groupids = config.groupids
test.timestamp = config.timestamp

Train = False
Predict = not Train

config.sequence_length = 1
config.batch_size = 1
config.shift = 1
config.restore_model = True
config.save_model = True
config.random = False
config.stateful = True
config.train_fids = [102]
config.test_fids = [202]

config.display()

m = TsLSTM(config)

if Train:
    m.train(train, test, config)


if Predict:
    dtrain = train #.loc[train.ENGINEID<9]
    y_target = dtrain["RUL"]
    y_pred = m.predict(dtrain, config)
    z=pd.DataFrame(dict(target=y_target, pred=y_pred)).reset_index()
    #plt.plot(z[["target", "pred"]])

print ("df len %, predict len %" % (len(df[df.FILEID==102]), len(z)))

print(z.tail())
