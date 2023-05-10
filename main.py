from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import recommend, chat, model
from dependencies import get_token_header
from internal import admin

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

app.include_router(recommend.router, prefix="/recommend", tags=["recommend"],)
app.include_router(chat.router, prefix="/chat", tags=["chat"],)
app.include_router(model.router, prefix="/model", tags=["model"],)
app.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_token_header)],
    responses={418: {"description": "I'm a teapot"}},
)
