FROM python:3.12-slim

WORKDIR /app

# Context is now ./mem0 (submodule root)

# Create a staging directory for building the mem0 library
WORKDIR /app/mem0_build_stage

# Copy essential files for building mem0 from the submodule root
COPY pyproject.toml poetry.lock* README.md LICENSE ./
# The poetry.lock* will copy poetry.lock if it exists, otherwise ignore if it doesn't.
# If you don't use poetry.lock with pip install -e, you can omit it.

# Copy the actual mem0 library source code into a subdirectory named 'mem0'
# This makes the structure suitable for `pip install -e .` from /app/mem0_build_stage
COPY ./mem0 ./mem0/

# Install the local mem0 library in editable mode
RUN pip install --no-cache-dir -e .

# Go back to /app for the FastAPI server files
WORKDIR /app

# Copy server-specific requirements (which no longer includes mem0ai)
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Doppler CLI
RUN apt-get update && apt-get install -y curl && curl -Ls https://cli.doppler.com/install.sh | sh

# Copy wait-for-it.sh from server/wait-for-it.sh
COPY server/wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Copy the FastAPI wrapper (main.py) and any other files from server/ into /app/
# (main.py from mem0/server/main.py will be at /app/main.py)
COPY server/. /app/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
