FROM debian:bookworm-slim AS builder
WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    libfftw3-dev \
    libjpeg-dev \
    libopenblas-dev \
    libssl-dev \
    pkg-config \
    zlib1g-dev

ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . $HOME/.cargo/env && \
    rustup default stable

FROM builder AS backend-builder
ARG LIBTORCH
ARG LIBTORCH_LIB
ARG LD_LIBRARY_PATH
ARG CXXFLAGS
ARG CARGO_JOBS

ENV LIBTORCH=${LIBTORCH}
ENV LIBTORCH_LIB=${LIBTORCH_LIB}
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
ENV CXXFLAGS=${CXXFLAGS}

COPY libtorch /opt/libtorch
COPY Cargo.toml Cargo.lock ./
COPY backend ./backend
COPY frontend ./frontend
COPY shared ./shared
COPY pyproject/dfresearch /usr/src/app/pyproject/dfresearch
COPY pyproject/models /usr/src/app/pyproject/models

RUN --mount=type=secret,id=dburl \
    export DATABASE_URL=$(cat /run/secrets/dburl) && \
    . $HOME/.cargo/env && cd backend && cargo build --release -p backend

FROM backend-builder AS frontend-builder
RUN . $HOME/.cargo/env && rustup target add wasm32-unknown-unknown && cargo install trunk

COPY Cargo.toml Cargo.lock ./
COPY backend ./backend
COPY frontend ./frontend
COPY shared ./shared

RUN cd frontend && trunk build --release

FROM debian:bookworm-slim AS final
WORKDIR /usr/src/app

ARG LIBTORCH
ARG LIBTORCH_LIB
ARG LD_LIBRARY_PATH
ARG CXXFLAGS

ENV LIBTORCH=${LIBTORCH}
ENV LIBTORCH_LIB=${LIBTORCH_LIB}
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
ENV CXXFLAGS=${CXXFLAGS}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    libfftw3-double3 \
    libgcc-s1 \
    libjpeg62-turbo \
    libopenblas0 \
    libssl-dev \
    libstdc++6 \
    zlib1g

RUN curl https://pkg.cloudflareclient.com/pubkey.gpg | gpg --dearmor -o /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg] https://pkg.cloudflareclient.com/ $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/cloudflare-client.list && \
    apt-get update && apt-get install -y cloudflare-warp && \
    rm -rf /var/lib/apt/lists/*

COPY libtorch /opt/libtorch
COPY Cargo.toml Cargo.lock /usr/src/app/
COPY --from=frontend-builder /usr/src/app/frontend/ ./frontend
COPY --from=backend-builder /usr/src/app/target/release/backend /usr/local/bin/backend
COPY --from=backend-builder /usr/src/app/frontend/ ./frontend
COPY --from=backend-builder /usr/src/app/backend ./backend
COPY --from=backend-builder /usr/src/app/pyproject/models /usr/src/app/pyproject/models
COPY --from=backend-builder /usr/src/app/pyproject/dfresearch /usr/src/app/pyproject/dfresearch

EXPOSE 8081
CMD ["/usr/local/bin/backend"]
