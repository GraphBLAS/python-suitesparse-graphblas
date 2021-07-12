if [ $# -eq 0 ]
    then
        echo "Usage: ./docker_build.sh SUITESPARSE_BRANCH VERSION [BRANCH LOCATION PUSH]

Example: ./docker_build.sh v5.1.3 5.1.3.1 main clone push

If location is clone then a fresh git clone will be used.  
If push is provided then the script will attempt to push to dockerhub."
        exit 1
fi

IMAGE=graphblas/python-suitesparse-graphblas
SUITESPARSE=$1
VERSION=$2
BRANCH=$3
LOCATION=$4
PUSH=$5

COMPACT=${COMPACT:-0}

if [ "$LOCATION" = "clone" ]
then
    TMPDIR=$(mktemp -d)
    if [ ! -e $TMPDIR ]; then
        >&2 echo "Failed to create temp directory"
        exit 1
    fi
    trap "exit 1"           HUP INT PIPE QUIT TERM
    trap 'rm -rf "$TMPDIR"' EXIT
    
    cd $TMPDIR
    git clone --branch $BRANCH https://github.com/GraphBLAS/python-suitesparse-graphblas.git
    cd python-suitesparse-graphblas
fi

docker build \
       --build-arg SUITESPARSE=${SUITESPARSE} \
       --build-arg VERSION=${VERSION} \
       --build-arg COMPACT=${COMPACT} \
       -t $IMAGE:$VERSION \
       .

docker tag $IMAGE:$VERSION $IMAGE:latest

if [ "$PUSH" = "push" ]
then
    docker push $IMAGE:$VERSION
    docker push $IMAGE:latest
fi
