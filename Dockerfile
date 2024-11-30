FROM python:3.9-slim

WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    build-essential \
    python3-dev \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Install bazel
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg && \
    mv bazel.gpg /etc/apt/trusted.gpg.d/ && \
    echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    apt-get update && apt-get install -y bazel && \
    rm -rf /var/lib/apt/lists/*

# Install TensorFlow build dependencies
RUN pip install -U pip six numpy wheel setuptools mock future>=0.17.1 && \
    pip install -U keras_applications==1.0.6 --no-deps && \
    pip install -U keras_preprocessing==1.0.5 --no-deps

# Clone and build TensorFlow with automated configuration
RUN git clone https://github.com/tensorflow/tensorflow && \
    cd tensorflow && \
    git checkout r2.15 && \
    # Set environment variables for automated configuration
    export PYTHON_BIN_PATH=$(which python3) && \
    export PYTHON_LIB_PATH=/usr/local/lib/python3.9/site-packages && \
    export TF_NEED_ROCM=0 && \
    export TF_NEED_CUDA=0 && \
    export TF_NEED_CLANG=1 && \
    export CLANG_COMPILER_PATH=$(which clang) && \
    export TF_ENABLE_XLA=0 && \
    export CC_OPT_FLAGS="-Wno-sign-compare" && \
    export TF_SET_ANDROID_WORKSPACE=0 && \
    # Run configure with the environment variables
    ./configure && \
    # Build TensorFlow
    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 //tensorflow/tools/pip_package:build_pip_package && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg && \
    pip install /tmp/tensorflow_pkg/tensorflow-*.whl && \
    cd .. && rm -rf tensorflow

# Copy requirements first (excluding tensorflow since we built it)
COPY requirements.txt .
RUN grep -v tensorflow requirements.txt > requirements_no_tf.txt && \
    pip install --no-cache-dir -r requirements_no_tf.txt

# Copy the application code
COPY api /app/api

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]