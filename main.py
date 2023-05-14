from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import recommend, chat, model


def get_application():
    app = FastAPI(title="seoul-competition-ai", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080", "http://springboot:8080"],
        allow_methods=["get", "post", "put", "update"],
        allow_credentials=True,
        allow_headers=["*"],
    )
    return app


app = get_application()

app.include_router(recommend.router, prefix="/recommend", tags=["recommend"],)
app.include_router(chat.router, prefix="/chat", tags=["chat"],)
app.include_router(model.router, prefix="/model", tags=["model"],)
