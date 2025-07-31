import os
import pymysql


host = os.environ.get('crawler_host', 'localhost')  # Use environment variable or default to localhost)
port =  os.environ.get("crawler_port")
user = os.environ.get("crawler_user")
password =  os.environ.get("crawler_pass")
db_name =  os.environ.get("crawler_db")

connection = pymysql.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    db=db_name,
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

try:
    # Check if the 'newsarticle' table exists
    with connection.cursor() as cursor:
        table_name = 'newsarticle'
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        table_exists = cursor.fetchone()
    # If the table doesn't exist, create it
    if not table_exists:
        with connection.cursor() as cursor:
            
            create_table_query = """
         

          CREATE TABLE totals (
            id INT Auto_Increment PRIMARY KEY,
            entity VARCHAR(255) UNIQUE,
            score INT,
            type VARCHAR(255)
        );
            """
            
            cursor.execute(create_table_query)

        connection.commit()
finally:
    connection.close()        
