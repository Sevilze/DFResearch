ARG MODELS_IMAGE=scratch
FROM ${MODELS_IMAGE} AS models

FROM rust:1.87-bookworm AS planner
WORKDIR /usr/src/app
RUN cargo install cargo-chef
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM rust:1.87-bookworm AS builder
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
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN cargo install cargo-chef

ARG LIBTORCH
ARG LIBTORCH_LIB
ARG LD_LIBRARY_PATH
ARG CXXFLAGS
ARG CARGO_JOBS

ENV LIBTORCH=${LIBTORCH} \
    LIBTORCH_LIB=${LIBTORCH_LIB} \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
    CXXFLAGS=${CXXFLAGS} \
    CARGO_JOBS=${CARGO_JOBS}

COPY libtorch /opt/libtorch
COPY --from=planner /usr/src/app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY . .
RUN cd backend && cargo build --release --locked -p backend


FROM rust:1.87-bookworm AS frontend-builder
WORKDIR /usr/src/app
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rustup target add wasm32-unknown-unknown && \
    cargo install trunk

COPY Cargo.toml Cargo.lock ./
COPY backend ./backend
COPY frontend ./frontend
COPY shared ./shared

RUN cd frontend && npm install && npm run build-css-prod
RUN cd frontend && trunk build --release


FROM debian:bookworm-slim AS final
WORKDIR /usr/src/app

ARG LIBTORCH
ARG LIBTORCH_LIB
ARG LD_LIBRARY_PATH
ARG CXXFLAGS
ENV LIBTORCH=${LIBTORCH} \
    LIBTORCH_LIB=${LIBTORCH_LIB} \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
    CXXFLAGS=${CXXFLAGS}

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

COPY libtorch /opt/libtorch
COPY Cargo.toml Cargo.lock /usr/src/app/
COPY --from=frontend-builder /usr/src/app/frontend/ ./frontend
COPY --from=builder /usr/src/app/target/release/backend /usr/local/bin/backend
COPY --from=builder /usr/src/app/frontend/ ./frontend
COPY --from=builder /usr/src/app/backend ./backend
COPY --from=models /models/ /usr/src/app/pyproject/models/
COPY config /usr/src/app/config

EXPOSE 8081
CMD ["/usr/local/bin/backend"]
