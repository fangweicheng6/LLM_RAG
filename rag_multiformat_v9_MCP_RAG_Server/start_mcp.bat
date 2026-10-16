@echo off
rem MCP server launcher for the Industrial RAG project.
rem Default transport: streamable-http on http://127.0.0.1:8100/mcp (local only).
rem To accept LAN clients, add --host 0.0.0.0 (see the MCP handover doc in docs\ for firewall note).
rem For local stdio clients (Claude Desktop / Claude Code), pass --transport stdio,
rem or invoke deployment\mcp_server.py directly as documented in the docs\ MCP handover doc.
rem Keep this file ASCII-only: non-ASCII bytes confuse cmd.exe on cp936 consoles.
cd /d "%~dp0"
set "PY=D:\RAG\rag\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] venv python not found: %PY%
  echo Fix the PY path in this script, or start the server with a full interpreter path:
  echo   ^<python.exe^> -m deployment.mcp_server --transport streamable-http
  exit /b 1
)
echo.%*| findstr /c:"--transport" >nul
if errorlevel 1 goto default_http
"%PY%" -m deployment.mcp_server %*
goto done
:default_http
"%PY%" -m deployment.mcp_server --transport streamable-http %*
:done
