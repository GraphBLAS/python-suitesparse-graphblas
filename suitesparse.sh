
if [[ $1 =~ refs/tags/([0-9]\.[0-9]\.[0-9]).*$ ]];
then
    VERSION=${BASH_REMATCH[1]}
else
    exit -1

curl -L https://github.com/DrTimothyAldenDavis/GraphBLAS/archive/refs/tags/v${VERSION}.tar.gz | tar xzf -
cd GraphBLAS-${VERSION}/build
cmake .. -DGBCOMPACT=1
make -j$(nproc)
make install
