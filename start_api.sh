#!/bin/bash
# Start the Gomoku AI API Server
cd /home/z/my-project
export PATH="$HOME/.local/bin:$PATH"

# Kill existing
pkill -f "uvicorn api_server" 2>/dev/null
sleep 1

# Start server
nohup setsid /home/z/.local/bin/uvicorn api_server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --app-dir /home/z/my-project \
  --timeout-keep-alive 60 \
  > /home/z/my-project/api_server.log 2>&1 &

echo "API Server started. Check http://localhost:8000/api/health"
