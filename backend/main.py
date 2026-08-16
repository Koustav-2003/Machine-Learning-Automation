from fastapi import FastAPI

from backend.api.routes import router


app = FastAPI(
    title="Machine Learning Automation API",
    description="Backend API for automated machine learning.",
    version="1.0.0",
)


app.include_router(router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Machine Learning Automation API is running",
    }