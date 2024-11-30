FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow and other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir tensorflow-cpu==2.15.0 && \
    grep -v tensorflow requirements.txt > requirements_no_tf.txt && \
    pip install --no-cache-dir -r requirements_no_tf.txt && \
    rm requirements_no_tf.txt

# Copy the application code
COPY api /app/api

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]