#!/bin/sh
export MKL_NUM_THREADS=24;
export OMP_NUM_THREADS=24;
export LD_LIBRARY_PATH=/HOME/sysu_sc_ll/env/hdf5-new/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/HOME/sysu_sc_ll/WORKSPACE/wenyp/resnet-v9/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/HOME/sysu_sc_ll/intel/compilers_and_libraries/linux/mkl/lib/intel64:$LD_LIBRARY_PATH

set -e

TOOLS=./caffe-resnet/build/tools


yhrun -n 4 -N 4 -c 24 -p work  $TOOLS/caffe train \
  --solver=./solver_50.prototxt  2>&1 | tee train_50_4nodes.log
