#!/bin/bash
cd ~/Downloads/Private-Ai-Project
source venv/bin/activate.fish

# Start streamlit in background
streamlit run app.py &
STREAMLIT_PID=$!

echo "Streamlit started!"
sleep 3

# Start ngrok
ngrok http 8501 --request-header-add="ngrok-skip-browser-warning:true"

# When ngrok closes, kill streamlit too
kill $STREAMLIT_PID
