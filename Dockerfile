FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_HTTP_TIMEOUT=60

# Set working directory
WORKDIR /app

# Install uv package manager
RUN pip install uv

# Copy project files
COPY . .

# Install dependencies using uv
RUN uv add numpy
RUN uv add "mcp[cli]"

# Expose the port (if needed)
EXPOSE 8000

# Start the MCP server using uv
CMD ["uv", "run", "main.py"]