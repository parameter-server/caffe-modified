#!/usr/bin/env sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/HOME/sysu_sc_ll/env/hdf5-new/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/HOME/sysu_sc_ll/WORKSPACE/wenyp/resnet-ta/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/HOME/sysu_sc_ll/intel64
./build/tools/caffe train --solver=examples/mnist/lenet_solver_adam.prototxt \
  2>&1 | tee ./lenet_adam.log
