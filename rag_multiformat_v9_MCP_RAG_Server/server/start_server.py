import argparse
import uvicorn
from utils.config_loader import settings

def main():
    parser = argparse.ArgumentParser(description="RAG service launcher")
    parser.add_argument("--mode", choices=["socket", "http", "mcp", "mcp-http"], default="socket")
    args = parser.parse_args()

    if args.mode == "socket":
        from deployment.rag_server import start
        start()
    elif args.mode == "mcp":
        from deployment.mcp_server import main as mcp_main
        mcp_main(["--transport", "stdio"])
    elif args.mode == "mcp-http":
        # MCP streamable-http 默认端口/路径取自 service.mcp.*；
        # 需要自定义端口时直接用 python -m deployment.mcp_server --transport streamable-http --port <port>
        from deployment.mcp_server import main as mcp_main
        mcp_main(["--transport", "streamable-http"])
    else:
        uvicorn.run(
            "deployment.api_server:app",
            host=settings.get("service.server.http_host", "0.0.0.0"),
            port=int(settings.get("service.server.http_port", 8000))
        )

if __name__ == "__main__":
    main()
