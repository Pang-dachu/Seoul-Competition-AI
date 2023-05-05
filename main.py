from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import basic, recommend
from dependencies import get_token_header
from internal import admin
from db import get_mysql_connection


def get_application():
    app = FastAPI(title="Phresh", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = get_application()


@app.get("/")
async def read_root():
    # MySQL 연결 객체 생성
    conn = await get_mysql_connection()
    # 연결 객체에서 커서 객체 생성
    cursor = conn.cursor()
    # SQL 쿼리 실행
    cursor.execute("SELECT * FROM education LIMIT 10")
    # SQL 쿼리 결과 모두 가져오기
    result = cursor.fetchall()
    # 커서 객체와 연결 객체 닫기
    cursor.close(); conn.close()
    return {"db_result": result}

app.include_router(basic.router, prefix="/basic", tags=["basic"],)
app.include_router(recommend.router, prefix="/recommend", tags=["recommend"],)
app.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_token_header)],
    responses={418: {"description": "I'm a teapot"}},
)
