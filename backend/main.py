
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(
    title="SupportIQ AI API",
    description="AI-powered customer support ticket intelligence",
    version="1.0.0",
)

origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/")
def root():
    return {
        "name": "SupportIQ AI",
        "message": "SupportIQ AI API is running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "supportiq-ai-api",
    }
