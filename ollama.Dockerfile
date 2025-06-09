# Use a base image
FROM ubuntu:22.04

# Install curl
RUN apt-get update && apt-get install -y curl

# Download and install Ollama
RUN curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama && \
    chmod +x /usr/bin/ollama

# Expose the Ollama port
EXPOSE 11434

# Set the entrypoint to Ollama
ENTRYPOINT ["/usr/bin/ollama"] 