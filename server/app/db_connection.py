import datetime
import json
import csv
import psycopg2


with open("../server/env/id.txt", "r") as f:
    user = f.read()

    
with open("../server/env/pw.txt", "r") as f:
    password = f.read()

def connect_db():
    conn = psycopg2.connect(
        host="engineer.i4624.tk", # Server
        database="factory", # User & Default database
        user=user, # User & Default database
        password=password,  # Password
        port=50132 )# Port
    return conn
    



# # only to first try
# cur.execute("""CREATE TABLE test_return2 (
#     id serial PRIMARY KEY,  
#     infer_time timestamp,
#     RUL float,
#     RUL_time time
# );
# """)

def insert_data(conn, csv_path, current_time):
    
    cur = conn.cursor()  # cursor
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
            
        # Insert data into the database
        for row in csv_reader:
            prediction = row[0]  # Value from the first column
            timestamp = row[1]  # Value from the second column
            query = "INSERT INTO test_return2 (infer_time, RUL, RUL_time) VALUES (%s, %s, %s)"
            cur.execute(query, (current_time, prediction, timestamp))
            conn.commit()


    
def disconnect_db(conn):
    conn.close()
    
## using this to select 
# query = "SELECT * FROM test_return WHERE id = %s"
# cur.execute(query, (desired_id,))
# result = cur.fetchone()