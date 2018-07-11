#!/usr/bin/env sh
export LD_LIBRARY_PATH=/HOME/nsfc2015_569/env/hdf5-new/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/HOME/nsfc2015_569/intel/compilers_and_libraries/linux/mkl/lib/intel64:$LD_LIBRARY_PATH
TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver.prototxt

