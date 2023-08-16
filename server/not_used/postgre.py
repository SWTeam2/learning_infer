import psycopg2


with open("./env/id.txt", "r") as f:
    user = f.read()

    
with open("./env/pw.txt", "r") as f:
    password = f.read()


conn = psycopg2.connect(
    host="engineer.i4624.tk", # Server
    database="factory", # User & Default database
    user=user, # User & Default database
    password=password,  # Password
    port=50132 )# Port
    

cur = conn.cursor() # cursor

cur.execute("""CREATE TABLE test_return (
    RUL float,
    RUL_time timestamp,
    infer_time timestamp
);
""")

cur.execute("INSERT INTO test_table (name, age) VALUES ('spongebob', 12);")

conn.commit()