import argparse
import uvicorn
from utils.config_loader import settings

def main():
    parser = argparse.ArgumentParser(description="RAG service launcher")
    parser.add_argument("--mode", choices=["socket", "http"], default="socket")
    args = parser.parse_args()

    if args.mode == "socket":
        from deployment.rag_server import start
        start()
    else:
        uvicorn.run(
            "deployment.api_server:app",
            host=settings.get("service.server.http_host", "0.0.0.0"),
            port=int(settings.get("service.server.http_port", 8000))
        )

if __name__ == "__main__":
    main()
