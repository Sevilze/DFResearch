# Python Dependencies
    FROM rust:latest AS python-deps
    WORKDIR /usr/src/app
    
    RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        build-essential pkg-config \
        libjpeg-dev zlib1g-dev \
        libopenblas-dev libfftw3-dev \
        libstdc++6 libgomp1 \
        && rm -rf /var/lib/apt/lists/*
    
    ADD fsrequirements.txt /usr/src/app/fsrequirements.txt
    
    RUN --mount=type=cache,target=/root/.cache/pip \
        pip3 install --upgrade pip --break-system-packages && \
        pip3 install -r fsrequirements.txt \
            --extra-index-url https://download.pytorch.org/whl/cu126 \
            --break-system-packages
    
    
    # Build the Rust backend
    FROM python-deps AS backend-builder
    WORKDIR /usr/src/app
    
    RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        && rm -rf /var/lib/apt/lists/*
    
    ENV PYTHON_LIB_DIR=/usr/local/lib/python3.11/dist-packages
    ENV LIBTORCH=/usr/local/lib/python3.11/dist-packages/torch
    ENV LIBTORCH_LIB=/usr/local/lib/python3.11/dist-packages/torch
    ENV LIBTORCH_USE_PYTORCH=1
    ENV LD_LIBRARY_PATH=$LIBTORCH/lib:$PYTHON_LIB_DIR:$LD_LIBRARY_PATH
    ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
    ENV TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
    
    COPY libtorch /opt/libtorch/
    
    COPY Cargo.toml Cargo.lock ./
    COPY backend ./backend
    COPY frontend ./frontend
    COPY shared ./shared
    COPY pyproject/dfresearch /usr/src/app/pyproject/dfresearch
    COPY pyproject/models /usr/src/app/pyproject/models
    
    RUN cd backend && cargo build --release -p backend
    
    
    # Build the frontend
    FROM backend-builder AS frontend-builder
    WORKDIR /usr/src/app
    ENV CARGO_JOBS=1
    RUN rustup target add wasm32-unknown-unknown && \
        cargo install trunk
    
    COPY Cargo.toml Cargo.lock ./
    COPY backend ./backend
    COPY frontend ./frontend
    COPY shared ./shared
    
    ENV PYTHON_LIB_DIR=/usr/local/lib/python3.11/dist-packages
    ENV LIBTORCH=/usr/local/lib/python3.11/dist-packages/torch
    ENV LIBTORCH_LIB=/usr/local/lib/python3.11/dist-packages/torch
    ENV LIBTORCH_USE_PYTORCH=1
    ENV LD_LIBRARY_PATH=$LIBTORCH/lib:$PYTHON_LIB_DIR:$LD_LIBRARY_PATH
    ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
    ENV TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
    
    RUN cd frontend && trunk build --release
    
    
    # Final runtime image
    FROM debian:latest AS final
    WORKDIR /usr/src/app
    
    # Install runtime dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 libpython3.11 \
        libstdc++6 libgcc-s1 \
        libssl-dev ca-certificates \
        libgomp1 libjpeg62-turbo zlib1g \
        libopenblas0 libfftw3-double3 \
        && rm -rf /var/lib/apt/lists/*
    
    RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

    COPY Cargo.toml Cargo.lock /usr/src/app/
    COPY --from=python-deps /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
    COPY --from=backend-builder /opt/libtorch /opt/libtorch
    COPY --from=backend-builder /usr/src/app/target/release/backend /usr/local/bin/backend
    COPY --from=frontend-builder /usr/src/app/frontend/ ./frontend
    
    COPY --from=backend-builder /usr/src/app/backend ./backend
    COPY --from=backend-builder /usr/src/app/pyproject/models /usr/src/app/pyproject/models
    COPY --from=backend-builder /usr/src/app/pyproject/dfresearch /usr/src/app/pyproject/dfresearch
    
    ENV PYTHON_LIB_DIR=/usr/local/lib/python3.11/dist-packages
    ENV LIBTORCH=/usr/local/lib/python3.11/dist-packages/torch
    ENV LIBTORCH_LIB=/usr/local/lib/python3.11/dist-packages/torch
    ENV LIBTORCH_USE_PYTORCH=1
    ENV LD_LIBRARY_PATH=$LIBTORCH/lib:$PYTHON_LIB_DIR:$LD_LIBRARY_PATH
    ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
    ENV PYTHONPATH=/usr/src/app
    ENV TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
    ENV CXXFLAGS="-std=c++17"
    
    EXPOSE 8081
    CMD ["/usr/local/bin/backend"]
    