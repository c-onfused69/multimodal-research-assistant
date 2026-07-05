"""FastAPI application entrypoint."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.admin_routes import router as admin_router

app = FastAPI(title="Multimodal Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1/admin")


# Mount the static UI directory at the root and admin at /admin
app.mount("/admin", StaticFiles(directory="ui/admin", html=True), name="admin_ui")
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")


@app.get("/health")
def health_check():
    return {"status": "ok"}
