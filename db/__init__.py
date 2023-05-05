import mysql.connector


async def get_mysql_connection():
    conn = mysql.connector.connect(
        host="mysqldb",
        user="test_user",
        password="a8Gh@c8wi#gL^",
        database="userdb"
    )
    return conn
