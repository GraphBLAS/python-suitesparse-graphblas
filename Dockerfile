ARG BASE_CONTAINER=debian:sid-slim
FROM ${BASE_CONTAINER}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -yq build-essential cmake libpython3-dev python3-pip git

ARG RELEASE=v5.0.3
ARG COMPACT=0
ARG JOBS=1

WORKDIR /build
RUN git clone https://github.com/DrTimothyAldenDavis/GraphBLAS.git --depth 1 --branch $RELEASE
RUN git clone https://github.com/eliben/pycparser.git --depth 1

WORKDIR /build/GraphBLAS/build
RUN cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DGBCOMPACT=${COMPACT} \
    && make -j${JOBS} \
    && make install \
    && ldconfig

RUN pip3 install numpy cffi pytest cython

RUN mkdir -p /usr/local/lib/python3.9/dist-packages/pycparser/utils/fake_libc_include
RUN mv /build/pycparser/utils/fake_libc_include/* /usr/local/lib/python3.9/dist-packages/pycparser/utils/fake_libc_include/

RUN mkdir /build/python-suitesparse-graphblas
ADD . /build/python-suitesparse-graphblas
WORKDIR /build/python-suitesparse-graphblas
RUN git tag ${RELEASE}-docker-test-dont-ever-use
RUN python3 suitesparse_graphblas/create_headers.py
RUN python3 setup.py install
RUN ldconfig
RUN /bin/rm -Rf /build
WORKDIR /
RUN pytest --pyargs suitesparse_graphblas.tests
RUN apt-get -y --purge remove git python3-pip && apt-get clean
