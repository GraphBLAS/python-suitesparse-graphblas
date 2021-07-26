VERSION=${GITHUB_REF}
a=( ${version//./ } )
VERSION="${a[0]}.${a[1]}.${a[2]}"

curl -L https://github.com/DrTimothyAldenDavis/GraphBLAS/archive/refs/tags/v${VERSION}.tar.gz | tar xzf -
cd GraphBLAS-${VERSION}/build
cmake .. -DGBCOMPACT=1
make -j$(nproc)
make install
