import datetime
import json
import psycopg2


with open("../env/id.txt", "r") as f:
    user = f.read()

    
with open("../env/pw.txt", "r") as f:
    password = f.read()


conn = psycopg2.connect(
    host="engineer.i4624.tk", # Server
    database="factory", # User & Default database
    user=user, # User & Default database
    password=password,  # Password
    port=50132 )# Port
    

cur = conn.cursor() # cursor



# # only to first try
# cur.execute("""CREATE TABLE test_return2 (
#     id serial PRIMARY KEY,  
#     infer_time timestamp,
#     RUL float,
#     RUL_time time
# );
# """)



# Get the current time without microseconds
current_time = datetime.datetime.now().replace(microsecond=0)

data = {
    "RUL": 0.728,
    "RUL_time": datetime.datetime.strptime("8:13:01", "%H:%M:%S").time(),  # Convert to time object
    "infer_time": current_time  # Use current timestamp
}

# Extract specific fields from the JSON data
rul = data["RUL"]
rul_time = data["RUL_time"]
infer_time = data["infer_time"]

query = "INSERT INTO test_return2 (infer_time, RUL, RUL_time) VALUES (%s, %s, %s)"
cur.execute(query, (infer_time, rul, rul_time))

conn.commit()

## using this to select 
# query = "SELECT * FROM test_return WHERE id = %s"
# cur.execute(query, (desired_id,))
# result = cur.fetchone()