# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create .streamlit directory in the root user's home
RUN mkdir -p /root/.streamlit/

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Ensure the .streamlit directory exists in the app directory
RUN mkdir -p .streamlit

# Copy secrets file to both locations to ensure it's available
# First to the application directory
RUN if [ -f .streamlit/secrets.toml ]; then \
        cp .streamlit/secrets.toml /root/.streamlit/secrets.toml; \
    fi

# Expose port 8080 for Streamlit
EXPOSE 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
    CMD curl -f http://localhost:8080 || exit 1

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]