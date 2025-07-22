FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including tini and Doppler CLI
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    tini \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Doppler CLI
RUN wget -q -t3 'https://packages.doppler.com/public/cli/rsa.8004D9FF50437357.key' -O /etc/apt/keyrings/cli.gpg && \
    echo 'deb [signed-by=/etc/apt/keyrings/cli.gpg] https://packages.doppler.com/public/cli/deb/debian any-version main' | tee /etc/apt/sources.list.d/cli.list && \
    apt-get update && apt-get install -y doppler && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8002

# Run the application
CMD ["python", "main.py"]
