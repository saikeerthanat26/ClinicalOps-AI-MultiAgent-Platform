from fastapi import FastAPI

app = FastAPI(
    title="ClinicalOps AI",
    description="Local-first Multi-Agent Healthcare Intelligence Platform",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "application": "ClinicalOps AI",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "clinicalops-ai",
    }