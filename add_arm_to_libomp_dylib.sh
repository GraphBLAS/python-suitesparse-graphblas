#!/bin/sh

#mkdir x86lib
mkdir armlib

# download and unzip both x86 and arm libomp tarballs
#brew fetch --force --bottle-tag=x86_64_monterey libomp
brew fetch --force --bottle-tag=arm64_big_sur libomp

# untar
#tar -xzf $(brew --cache --bottle-tag=x86_64_monterey libomp) --strip-components 2 -C x86lib
tar -xzf $(brew --cache --bottle-tag=arm64_big_sur libomp) --strip-components 2 -C armlib

# merge
lipo armlib/lib/libomp.dylib $(brew --prefix libomp)/lib/libomp.dylib -output libomp.dylib -create
cp -f libomp.dylib $(brew --prefix libomp)/lib