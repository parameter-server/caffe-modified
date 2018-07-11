#!/bin/sh
export MKL_NUM_THREADS=24;
export OMP_NUM_THREADS=24;
export LD_LIBRARY_PATH=/HOME/sysu_sc_ll/env/hdf5-new/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/HOME/sysu_sc_ll/WORKSPACE/wenyp/resnet-v9/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/HOME/sysu_sc_ll/intel/compilers_and_libraries/linux/mkl/lib/intel64:$LD_LIBRARY_PATH

set -e

TOOLS=/HOME/sysu_sc_ll/WORKSPACE/wenyp/resnet-v9/caffe-resnet/build/tools

yhrun -n 1 -N 1 -c 24 -p work $TOOLS/caffe test \
  --model=./test20.prototxt  --weights=./cifar10_res20_4nodes_iter_64000_3.caffemodel --iterations=1000 2>&1 | tee ./test20_4nodes_iter64000_3.log
