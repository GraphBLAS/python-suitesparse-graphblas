ARG BASE_CONTAINER=python:3.12-slim-bookworm
FROM ${BASE_CONTAINER} AS suitesparse
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq build-essential cmake git

ARG SUITESPARSE
ARG COMPACT=0

WORKDIR /build
RUN git clone https://github.com/eliben/pycparser.git --depth 1

# Use `-DJITINIT=2` so that the JIT functionality is available, but disabled by default.
# Level 2, "run", means that pre-JIT kernels may be used, which does not require a compiler at runtime.
# Disable JIT entirely to avoid segfaulting in tests (matches CI in suitesparse.sh).
RUN git clone https://github.com/DrTimothyAldenDavis/GraphBLAS.git --depth 1 --branch ${SUITESPARSE} \
    && cd GraphBLAS/build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DCOMPACT=${COMPACT} \
        -DJITINIT=2 \
        -DGRAPHBLAS_USE_JIT=OFF \
    && make -j$(nproc) \
    && make install

FROM ${BASE_CONTAINER} AS psg
ARG VERSION
ENV PYTHONUNBUFFERED=1

COPY --from=suitesparse /usr/include/suitesparse/GraphBLAS.h /usr/local/include/suitesparse/GraphBLAS.h
# Copy only the real versioned library; recreate symlinks (Docker COPY collapses cross-stage symlinks).
COPY --from=suitesparse /usr/lib/x86_64-linux-gnu/libgraphblas.so.*.*.* /usr/lib/x86_64-linux-gnu/
RUN cd /usr/lib/x86_64-linux-gnu \
    && REAL=$(ls libgraphblas.so.*.*.*) \
    && SOMAJOR=libgraphblas.so.$(echo "$REAL" | sed -E 's/libgraphblas\.so\.([0-9]+).*/\1/') \
    && ln -sf "$REAL" "$SOMAJOR" \
    && ln -sf "$SOMAJOR" libgraphblas.so \
    && ldconfig

RUN apt-get update && apt-get install -yq build-essential git
RUN pip3 install --break-system-packages numpy cffi pytest cython pycparser setuptools wheel setuptools-git-versioning

COPY --from=suitesparse /build/pycparser/utils/fake_libc_include/* /usr/local/lib/python3.12/site-packages/pycparser/utils/fake_libc_include/

RUN mkdir -p /psg
ADD . /psg

WORKDIR /psg
# `git tag || true` so the build is idempotent when ${VERSION} already matches an existing tag in the source tree.
RUN (git tag ${VERSION} || true) && \
    python3 suitesparse_graphblas/create_headers.py && \
    pip3 install --break-system-packages --no-build-isolation --no-deps . && \
    ldconfig

#RUN pytest --pyargs suitesparse_graphblas.tests
RUN apt-get -y --purge remove git python3-pip && apt-get clean

FROM ${BASE_CONTAINER}
COPY --from=suitesparse /usr/lib/x86_64-linux-gnu/libgraphblas.so.*.*.* /usr/lib/x86_64-linux-gnu/
COPY --from=suitesparse /usr/lib/x86_64-linux-gnu/libgomp.so.*.*.* /usr/lib/x86_64-linux-gnu/
COPY --from=psg /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
RUN cd /usr/lib/x86_64-linux-gnu \
    && GBREAL=$(ls libgraphblas.so.*.*.*) \
    && GBSOMAJOR=libgraphblas.so.$(echo "$GBREAL" | sed -E 's/libgraphblas\.so\.([0-9]+).*/\1/') \
    && ln -sf "$GBREAL" "$GBSOMAJOR" \
    && ln -sf "$GBSOMAJOR" libgraphblas.so \
    && GMREAL=$(ls libgomp.so.*.*.*) \
    && GMSOMAJOR=libgomp.so.$(echo "$GMREAL" | sed -E 's/libgomp\.so\.([0-9]+).*/\1/') \
    && ln -sf "$GMREAL" "$GMSOMAJOR" \
    && ldconfig
