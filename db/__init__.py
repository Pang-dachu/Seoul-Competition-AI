import pymysql
import pymysql.cursors

async def get_mysql_connection():
    config = {
        "host":"mysqldb",
        "user":"test_user",
        "password":"a8Gh@c8wi#gL^",
        "database":"userdb",
        "charset":'utf8mb4',
        "connect_timeout":30,
        "read_timeout":30,
        "write_timeout":30,
        "cursorclass": pymysql.cursors.DictCursor  # 딕셔너리 형태로 결과를 반환하도록 설정
    }

    connection = pymysql.connect(**config)
    return connection
