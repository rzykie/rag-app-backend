#!/bin/bash
# This script starts the Ollama server and pulls the desired model.

# Start the Ollama server in the background
/usr/bin/ollama serve &

# Wait for the server to be ready
echo "Waiting for Ollama server to start..."
while ! curl -s http://localhost:11434 > /dev/null; do
    sleep 1
done
echo "Ollama server is ready."

# Pull the model specified in the config (or the default)
echo "Pulling model: ${LANGUAGE_MODEL:-qwen3:0.6b}"
/usr/bin/ollama pull "${LANGUAGE_MODEL:-qwen3:0.6b}"

# Keep the script running to keep the container alive
echo "Ollama is running with the model pulled. Tailing logs to keep container alive."
tail -f /dev/null 