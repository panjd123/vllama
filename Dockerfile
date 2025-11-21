# Use vLLM official image as base
FROM vllm/vllm-openai:latest

# Set working directory
WORKDIR /app

# Copy project files for installation
COPY pyproject.toml /app/
COPY README.md /app/
COPY vllama /app/vllama

# Install vllama package with all dependencies
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /root/.vllama/logs && \
    mkdir -p /root/.cache/huggingface/hub

# Expose vllama server port
EXPOSE 33258

# Expose vLLM instance port range
EXPOSE 33300-34300

# Run vllama serve - configuration is read from environment variables
ENTRYPOINT ["vllama", "serve"]
