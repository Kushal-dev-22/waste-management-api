# 1. Use official Python image
FROM python:3.12-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy requirements first for caching
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project into the container
COPY . .

# 6. Expose port 8000 (FastAPI default)
EXPOSE 8000

# 7. Command to run your FastAPI app with Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
