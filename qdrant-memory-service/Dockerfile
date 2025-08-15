FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including tini and Doppler CLI
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    tini \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Doppler CLI using official method
RUN (curl -Ls --tlsv1.2 --proto "=https" --retry 3 https://cli.doppler.com/install.sh || wget -t 3 -qO- https://cli.doppler.com/install.sh) | sh

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8002

# Run the application
CMD ["python", "main.py"]
