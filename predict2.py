from model import *
from config import *
from data_processing import *
today = datetime.date.today()

config = TsConf()

import psycopg2
conn = psycopg2.connect(dbname='elevator', user='elevator', 
                        password='elevator', host='95.213.193.6', port = '9432')
cursor = conn.cursor()

cursor.execute('SELECT * FROM PUBLIC.ELEVATOR')
records = pd.DataFrame(cursor.fetchall())
...
cursor.close()
conn.close()

print(type(records))
print(records[1,])
