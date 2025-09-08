from fastapi import FastAPI
import models
from database import engine
from routers import qa_router

# Create database tables
models.Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(title="RAG Q&A API")

# Include Routers
app.include_router(qa_router.router, tags=["Q&A"])

# Run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8888, reload=True)
