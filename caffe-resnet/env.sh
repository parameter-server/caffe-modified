#!/bin/bash
module load protobuf/3.1.0  leveldb/1.15.0  gflags/2.1.1 glog/0.3.3
module load myopencv myhdf5
# module load intel-compilers/2018
export LD_LIBRARY_PATH=/WORK/app/intel/parallel_studio_xe_2018/mkl/lib/intel64/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../lib
#export LIBRARY_PATH=/WORK/app/boost/1_54_0-icc-MPI/lib:$LIBRARY_PATH
