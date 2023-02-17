#!/bin/sh
# Construct a universal2 version of homebrew's libomp.
#
# Homebrew's libomp works well to patch Apple clang's missing OpenMP support. The problem is a combination of:
# - Brew installs libomp built for x86 *or* ARM, matching the architecture of the machine it is running on.
# - GitHub Actions only has x86 runners as of now. Check back in Q4 2023. https://github.com/github/roadmap/issues/528
# - The linker will select the first found libomp, and if that version does not include the expected architecture then
#   linking will fail.
#
# One solution is to build a universal2 version of libomp that includes both architectures. That's what this script
# does. It adds the ARM version of libomp to the x86 version.
#
# This script assumes it is running on x86 with x86 libomp already installed.

if [ "$(arch)" != "x86_64" ] && [ "$(arch)" != "i386" ]; then
    echo "Not running on x86 as expected. Running on:"
    arch
    echo "If the above says arm64 then this hack is no longer necessary. Remove this script from the build."
    exit 1;
fi

#mkdir x86lib
mkdir armlib

# download and unzip both x86 and arm libomp tarballs
#brew fetch --force --bottle-tag=x86_64_monterey libomp
brew fetch --force --bottle-tag=arm64_big_sur libomp

# untar
#tar -xzf $(brew --cache --bottle-tag=x86_64_monterey libomp) --strip-components 2 -C x86lib
tar -xzf $(brew --cache --bottle-tag=arm64_big_sur libomp) --strip-components 2 -C armlib

# ARM and x86 dylibs have different install names due to different brew install directories.
# The x86 install name will be expected so make the ARM install name match.
X86_INSTALL_NAME="$(otool -X -D $(brew --prefix libomp)/lib/libomp.dylib)"
install_name_tool -id "${X86_INSTALL_NAME}" armlib/lib/libomp.dylib
codesign --force -s - armlib/lib/libomp.dylib

# merge the downloaded (arm) libomp with the already installed (x86) libomp to create a universal libomp
lipo armlib/lib/libomp.dylib $(brew --prefix libomp)/lib/libomp.dylib -output libomp.dylib -create

# print contents of universal library for reference
otool -arch all -L libomp.dylib

# replace the x86-only libomp with the newly-created universal one
cp -f libomp.dylib $(brew --prefix libomp)/lib

# clean up
rm libomp.dylib
rm -rf armlib
