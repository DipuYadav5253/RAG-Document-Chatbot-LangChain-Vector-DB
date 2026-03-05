# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Create necessary folders
RUN mkdir -p docs faiss_index mlruns

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT 
