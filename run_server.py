import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    print(f"📖 Swagger Documentation: http://localhost:{settings.PORT}/docs")
    print(f"🌐 Alternative ReDoc: http://localhost:{settings.PORT}/redoc")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
