FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including tini
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8002

# Run the application
CMD ["python", "main.py"]
