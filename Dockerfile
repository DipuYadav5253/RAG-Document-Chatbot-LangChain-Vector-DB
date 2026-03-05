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

# Expose any port (Render overrides with $PORT)
EXPOSE 8000

# Run FastAPI with dynamic port
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]