ARG BASE_CONTAINER=python:3.9-slim-buster
FROM ${BASE_CONTAINER} as suitesparse

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -yq build-essential cmake git

ARG SUITESPARSE=v5.0.3
ARG COMPACT=1

WORKDIR /build
RUN git clone https://github.com/eliben/pycparser.git --depth 1

WORKDIR /build/GraphBLAS/build
RUN git clone https://github.com/DrTimothyAldenDavis/GraphBLAS.git --depth 1 --branch $SUITESPARSE \
    && cd GraphBLAS/build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DGBCOMPACT=${COMPACT} \
    && make -j$(nproc) \
    && make install

FROM ${BASE_CONTAINER} as psg
ARG SUITESPARSE=v5.0.3
ENV PYTHONUNBUFFERED 1

COPY --from=suitesparse /usr/lib/x86_64-linux-gnu/libgraphblas* /usr/lib/x86_64-linux-gnu/
COPY --from=suitesparse  /usr/include/GraphBLAS.h /usr/local/include/
COPY --from=suitesparse /build/pycparser/utils/fake_libc_include/* /usr/local/lib/python3.9/site-packages/pycparser/utils/fake_libc_include/

RUN apt-get update && apt-get install -yq build-essential git
RUN pip3 install numpy cffi pytest cython
    
RUN mkdir -p /build/python-suitesparse-graphblas
ADD . /build/python-suitesparse-graphblas

WORKDIR /build/python-suitesparse-graphblas
RUN git tag ${SUITESPARSE}.0-do-no-use && \
    python3 suitesparse_graphblas/create_headers.py && \
    python3 setup.py install && \
    ldconfig

WORKDIR /
RUN pytest --pyargs suitesparse_graphblas.tests
RUN apt-get -y --purge remove git python3-pip && apt-get clean

FROM ${BASE_CONTAINER}
COPY --from=suitesparse /usr/lib/x86_64-linux-gnu/libgraphblas* /usr/lib/x86_64-linux-gnu/
COPY --from=suitesparse /usr/lib/x86_64-linux-gnu/libgomp* /usr/lib/x86_64-linux-gnu/
COPY --from=psg /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
