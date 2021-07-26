curl -Ls https://github.com/DrTimothyAldenDavis/GraphBLAS/archive/refs/tags/v5.1.5.tar.gz | tar xzf -
cd GraphBLAS-5.1.5/build
cmake .. -DGBCOMPACT=1
make -j$(nproc)
sudo make install
