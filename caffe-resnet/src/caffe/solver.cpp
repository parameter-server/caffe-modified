#include <cstdio>
#include <cstdlib>

#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <boost/thread.hpp>
#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
  : net_(), callbacks_(), root_solver_(root_solver),
    requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
  : net_(), callbacks_(), root_solver_(root_solver),
    requested_early_exit_(false) {
  SolverParameter param;
  ReadProtoFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>//modified 1
void Solver<Dtype>::Init(const SolverParameter& param) {
  MPI_Init(NULL,NULL);
  //LOG(INFO) << " signal 1_1";
  int rank;
  MPI_Comm c_comm;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &c_comm);

  //LOG(INFO) << " signal 1_2";
  CHECK(Caffe::root_solver() || root_solver_)
      << "root_solver_ needs to be set for all non-root solvers";
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
                                     << std::endl << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  if (Caffe::root_solver() && param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  if (Caffe::root_solver()) {
    InitTestNets();
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
                             param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
                              << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
                              << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  if (Caffe::root_solver()) {
    net_.reset(new Net<Dtype>(net_param));
  } else {
    net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  CHECK(Caffe::root_solver());
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
    CHECK_GE(param_.test_iter_size(), num_test_nets)
        << "test_iter must be specified for each test network.";
  } else {
    CHECK_EQ(param_.test_iter_size(), num_test_nets)
        << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
    sources[test_net_id] = "test_net_param";
    net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
    sources[test_net_id] = "test_net file: " + param_.test_net(i);
    ReadNetParamsFromTextFileOrDie(param_.test_net(i),
                                   &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    if (Caffe::root_solver()) {
      test_nets_[i].reset(new Net<Dtype>(net_params[i]));
      LOG(INFO) << "Test network set";
    } else {
      test_nets_[i].reset(new Net<Dtype>(net_params[i],
                                         root_solver_->test_nets_[i].get()));
    }
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <typename Dtype>//added 1
void Solver<Dtype>::SyncParameter() {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  for (int param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {
    int count = net_params[param_id]->count();
    Dtype* params_diff = net_params[param_id]->mutable_cpu_data();
    MPI_Op op = (sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Bcast(params_diff, count, op, 0, MPI_COMM_WORLD);
    //MPI_Barrier(MPI_COMM_WORLD);
  }
  LOG(INFO) << "parameter sync set";
}

template <typename Dtype>//added 2
void Solver<Dtype>::Display(const Dtype smoothed_loss) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //  MPI_Op op = (sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE;
   FLAGS_alsologtostderr = 1;
  float lapse = iteration_timer_.Seconds();
  float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
  LOG_IF(INFO, Caffe::root_solver()) << " rank = " << rank
																		 << " Iteration " << iter_
      << " (" << per_s << " iter/s, " << lapse << "s/"
      << param_.display() << " iters), loss = " << smoothed_loss;
  iteration_timer_.Start();
  iterations_last_ = iter_;
  const vector<Blob<Dtype>*>& result = net_->output_blobs();
  int score_index = 0;
  for (int j = 0; j < result.size(); ++j) {
    const Dtype* result_vec = result[j]->cpu_data();
    const string& output_name =
      net_->blob_names()[net_->output_blob_indices()[j]];
    const Dtype loss_weight =
      net_->blob_loss_weights()[net_->output_blob_indices()[j]];
    for (int k = 0; k < result[j]->count(); ++k) {
      ostringstream loss_msg_stream;
      if (loss_weight) {
        loss_msg_stream << " (* " << loss_weight
                        << " = " << loss_weight * result_vec[k] << " loss)";
      }
      LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
                                         << score_index++ << ": " << output_name << " = "
                                         << result_vec[k] << loss_msg_stream.str();
    }
  }
  if (rank != 0)
     FLAGS_alsologtostderr = 0;
}

template <typename Dtype>//added 3
void Solver<Dtype>::AllReduceSumBuffer(Dtype *buffer1, Dtype *buffer2, int buffe_size, int iter) {
  MPI_Op op = (sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE;
  MPI_Allreduce(buffer1, buffer2, buffe_size, op, MPI_SUM, MPI_COMM_WORLD);
  LOG(INFO) << "Iter " << iter << " transfer Done";
}

template <typename Dtype>//modified 2
void Solver<Dtype>::Step(int iters) {
  //LOG(INFO) << "Iteration begin";
  int rank = 0, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Op op = (sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE;

  vector<Blob<Dtype>*> bottom_vec;
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  vector<Dtype> losses;
  Dtype smoothed_loss = 0;
  Dtype aver_loss = 0;

  int para_buffer_size = 0;
  int* para_index = new int [this->net_->learnable_params().size()];
  const vector<Blob<Dtype>*>& net_params_ = this->net_->learnable_params();
  for (int param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {
    para_index[param_id] = para_buffer_size;
    para_buffer_size += net_params_[param_id]->count();
  }
  para_buffer_size = para_buffer_size + 1;

  LOG(INFO) << "Ready to sync parameter"
            << " numbers of parameter " << para_buffer_size
            << " numbers of parameter blob " << this->net_->learnable_params().size();
  SyncParameter();

  Dtype *para_buffer ;
  Dtype *para_buffer_;
  Dtype *para_buffer_pre;
  Dtype *para_buffer1 = new Dtype[sizeof(Dtype)*para_buffer_size];
  Dtype *para_buffer2 = new Dtype[sizeof(Dtype)*para_buffer_size];
  Dtype *para_buffer3 = new Dtype[sizeof(Dtype)*para_buffer_size];
  Dtype *para_buffer4 = new Dtype[sizeof(Dtype)*para_buffer_size];

  para_buffer = para_buffer1;
  para_buffer_ = para_buffer2;
  CPUTimer collect_timer, copy_timer, total_timer, wait_timer;
  // caffe_set(para_buffer_size, (Dtype)100.0, para_buffer1);
  // caffe_set(para_buffer_size, (Dtype)100.0, para_buffer2);
  // caffe_set(para_buffer_size, (Dtype)100.0, para_buffer3);
  // caffe_set(para_buffer_size, (Dtype)100.0, para_buffer4);
  // char csb[10];
  // string parafile = "pararecordfile";
  // std::sprintf(csb, "%d", rank);
  // parafile = parafile + string(csb) + ".txt";
  // std::ofstream file(parafile.c_str(), std::ios::out | std::ios::ate);
  MPI_Barrier(MPI_COMM_WORLD);
  CPUTimer barr_timer, iter_timer, part_timer;

  while (iter_ < stop_iter) {
    iter_timer.Start();
    //FLAGS_alsologtostderr = 1;
    //LOG(INFO) << "Iteration " << iter_ << " begins";
    // if (rank != 0) {
    //   FLAGS_alsologtostderr = 0;
    // }
    // zero-init the params
    net_->ClearParamDiffs();
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())
        && Caffe::root_solver()) {
      TestAll();
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }

    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    Dtype loss = 0;
    for (int i = 0; i < param_.iter_size(); ++i) {
      loss += net_->ForwardBackward(bottom_vec);
    }
    loss = loss / param_.iter_size();
    // average the loss across iterations for smoothed reporting

    if (losses.size() < average_loss) {
      losses.push_back(loss);
      int size = losses.size();
      smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
    } else {
      int idx = (iter_ - start_iter) % average_loss;
      smoothed_loss += (loss - losses[idx]) / average_loss;
      losses[idx] = loss;
    }

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }

    part_timer.Start();
    copy_timer.Start();
    const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
    #pragma omp parallel for num_threads(24)
    for (int param_id = 0; param_id < this->net_->learnable_params().size();
         ++param_id) {
      Dtype* params_diff = net_params[param_id]->mutable_cpu_diff();
      caffe_copy(net_params_[param_id]->count(), params_diff, para_buffer + para_index[param_id]);
    }
    para_buffer[para_buffer_size - 1] = smoothed_loss;
    copy_timer.Stop();

    barr_timer.Start();
    MPI_Barrier(MPI_COMM_WORLD);
    barr_timer.Stop();

    collect_timer.Start();
    MPI_Allreduce(para_buffer, para_buffer_, para_buffer_size, op, MPI_SUM, MPI_COMM_WORLD);
    collect_timer.Stop();

    total_timer.Start();
    #pragma omp parallel for num_threads(24)
    for (int param_id = 0; param_id < this->net_->learnable_params().size();
         ++param_id) {
      Dtype* params_diff = net_params[param_id]->mutable_cpu_diff();
      caffe_copy(net_params_[param_id]->count(), para_buffer_ + para_index[param_id], params_diff);
    }
    aver_loss = para_buffer_[para_buffer_size - 1] / mpi_size;
    total_timer.Stop();

    //LOG(INFO) << iter_ << "  "
    //          << " copy gradient times " << copy_timer.MilliSeconds() << " ms."
    //          << " barrier times " << barr_timer.MilliSeconds() << " ms."
    //          << " Collect gradient times " << collect_timer.MilliSeconds() << " ms."
    //          << " copy gradient times " << total_timer.MilliSeconds() << " ms.";

    part_timer.Stop();

    if (display) {
      CPUTimer display_time;
      display_time.Start();
      boost::thread disp(boost::bind(&Solver<Dtype>::Display, this, smoothed_loss));
      // disp.detach();
      display_time.Stop();
      // seems to affect the loss, maybe used only when real train
      LOG(INFO) << iter_ << "  "
                << "display time " << display_time.MilliSeconds() << " ms.";
    }
    if (iter_ != start_iter)
      ApplyUpdate();

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
        (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
    iter_timer.Stop();
    LOG(INFO) << "iteraion " << iter_ - 1 << " cost " << iter_timer.MilliSeconds() << " ms totally"
              << "certain part cost" << part_timer.MilliSeconds();
  }
  if (para_buffer1 != NULL)
    delete [] para_buffer1;
  if (para_buffer2 != NULL)
    delete [] para_buffer2;
  if (para_buffer3 != NULL)
    delete [] para_buffer3;
  if (para_buffer4 != NULL)
    delete [] para_buffer4;
  if (para_index != NULL)
    delete [] para_index;
  //file.close();
}

template <typename Dtype>//modified 3?
void Solver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  Step(param_.max_iter() - iter_);

  MPI_Barrier(MPI_COMM_WORLD);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    Dtype loss;
    net_->ForwardPrefilled(&loss);
    LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>//modified 4?
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
  ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
      if (SolverAction::SNAPSHOT == request) {
        Snapshot();
      } else if (SolverAction::STOP == request) {
        requested_early_exit_ = true;
      }
      request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
      test_net->Forward(bottom_vec, &iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  int rank, mpi_size;

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //if (rank == 0)
  // FLAGS_alsologtostderr = 1;

  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
      test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output local #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
    Dtype all_score = 0;
    MPI_Op op = (sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Reduce(&mean_score, &all_score, 1,  op, MPI_SUM, 0,
           MPI_COMM_WORLD);
    if (rank == 0) {
      all_score /= mpi_size;
      LOG(INFO) << "    Test net output all #" << i << ": " << output_name << " = "
                << all_score << loss_msg_stream.str();
    }
  }
  // if (rank == 0)
  //   FLAGS_alsologtostderr = 1;
  // else
  //   FLAGS_alsologtostderr = 0;
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>//modified 5
string Solver<Dtype>::SnapshotFilename(const string extension) {
  string filename(param_.snapshot_prefix());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const int kBufferSize = 20;
  char iter_str_buffer[kBufferSize];
  snprintf(iter_str_buffer, kBufferSize, "_iter_%d_%d", iter_, rank);
  return filename + iter_str_buffer + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
//template <typename Dtype>//added 4
//Dtype SGDSolver<Dtype>::GetLearningRate() {
//  Dtype rate;
//  const string& lr_policy = this->param_.lr_policy();
//  if (lr_policy == "fixed") {
//    rate = this->param_.base_lr();
//  } else if (lr_policy == "step") {
//    this->current_step_ = this->iter_ / this->param_.stepsize();
//    rate = this->param_.base_lr() *
//           pow(this->param_.gamma(), this->current_step_);
//  } else if (lr_policy == "exp") {
//    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
//  } else if (lr_policy == "inv") {
//    rate = this->param_.base_lr() *
//           pow(Dtype(1) + this->param_.gamma() * this->iter_,
//               - this->param_.power());
//  } else if (lr_policy == "multistep") {
//    if (this->current_step_ < this->param_.stepvalue_size() &&
//        this->iter_ >= this->param_.stepvalue(this->current_step_)) {
//      this->current_step_++;
//      LOG(INFO) << "MultiStep Status: Iteration " <<
//                this->iter_ << ", step = " << this->current_step_;
//    }
//    rate = this->param_.base_lr() *
//           pow(this->param_.gamma(), this->current_step_);
//  } else if (lr_policy == "poly") {
//    rate = this->param_.base_lr() * pow(Dtype(1.) -
//                                        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
//                                        this->param_.power());
//  } else if (lr_policy == "sigmoid") {
//    rate = this->param_.base_lr() * (Dtype(1.) /
//                                     (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
//                                         Dtype(this->param_.stepsize())))));
//  } else {
//    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
//  }
//  return rate;
//}
//
//template <typename Dtype>//added 5
//void SGDSolver<Dtype>::PreSolve() {
//  // Initialize the history
//  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
//  history_.clear();
//  update_.clear();
//  temp_.clear();
//  for (int i = 0; i < net_params.size(); ++i) {
//    const vector<int>& shape = net_params[i]->shape();
//    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
//    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
//    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
//  }
//}
//
//template <typename Dtype>//added 6
//void SGDSolver<Dtype>::ClipGradients() {
//  const Dtype clip_gradients = this->param_.clip_gradients();
//  if (clip_gradients < 0) { return; }
//  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
//  Dtype sumsq_diff = 0;
//  for (int i = 0; i < net_params.size(); ++i) {
//    sumsq_diff += net_params[i]->sumsq_diff();
//  }
//  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
//  if (l2norm_diff > clip_gradients) {
//    Dtype scale_factor = clip_gradients / l2norm_diff;
//    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
//              << l2norm_diff << " > " << clip_gradients << ") "
//              << "by scale factor " << scale_factor;
//    for (int i = 0; i < net_params.size(); ++i) {
//      net_params[i]->scale_diff(scale_factor);
//    }
//  }
//}
//
//template <typename Dtype>//added 7
//void SGDSolver<Dtype>::ApplyUpdate() {
//  CHECK(Caffe::root_solver());
//  Dtype rate = GetLearningRate();
//  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
//    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
//  }
//  ClipGradients();
//  for (int param_id = 0; param_id < this->net_->learnable_params().size();
//       ++param_id) {
//    Normalize(param_id);
//    Regularize(param_id);
//    ComputeUpdateValue(param_id, rate);
//  }
//  this->net_->Update();
//}
//
//template <typename Dtype>//added 8
//void SGDSolver<Dtype>::Normalize(int param_id) {
//  if (this->param_.iter_size() == 1) { return; }
//  // Scale gradient to counterbalance accumulation.
//  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
//  const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
//  switch (Caffe::mode()) {
//  case Caffe::CPU: {
//    caffe_scal(net_params[param_id]->count(), accum_normalization,
//               net_params[param_id]->mutable_cpu_diff());
//    break;
//  }
//  case Caffe::GPU: {
//#ifndef CPU_ONLY
//    caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
//                   net_params[param_id]->mutable_gpu_diff());
//#else
//    NO_GPU;
//#endif
//    break;
//  }
//  default:
//    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
//  }
//}
//
//template <typename Dtype>//added 9
//void SGDSolver<Dtype>::Regularize(int param_id) {
//  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
//  const vector<float>& net_params_weight_decay =
//    this->net_->params_weight_decay();
//  Dtype weight_decay = this->param_.weight_decay();
//  string regularization_type = this->param_.regularization_type();
//  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
//  switch (Caffe::mode()) {
//  case Caffe::CPU: {
//    if (local_decay) {
//      if (regularization_type == "L2") {
//        // add weight decay
//        caffe_axpy(net_params[param_id]->count(),
//                   local_decay,
//                   net_params[param_id]->cpu_data(),
//                   net_params[param_id]->mutable_cpu_diff());
//      } else if (regularization_type == "L1") {
//        caffe_cpu_sign(net_params[param_id]->count(),
//                       net_params[param_id]->cpu_data(),
//                       temp_[param_id]->mutable_cpu_data());
//        caffe_axpy(net_params[param_id]->count(),
//                   local_decay,
//                   temp_[param_id]->cpu_data(),
//                   net_params[param_id]->mutable_cpu_diff());
//      } else {
//        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
//      }
//    }
//    break;
//  }
//  case Caffe::GPU: {
//#ifndef CPU_ONLY
//    if (local_decay) {
//      if (regularization_type == "L2") {
//        // add weight decay
//        caffe_gpu_axpy(net_params[param_id]->count(),
//                       local_decay,
//                       net_params[param_id]->gpu_data(),
//                       net_params[param_id]->mutable_gpu_diff());
//      } else if (regularization_type == "L1") {
//        caffe_gpu_sign(net_params[param_id]->count(),
//                       net_params[param_id]->gpu_data(),
//                       temp_[param_id]->mutable_gpu_data());
//        caffe_gpu_axpy(net_params[param_id]->count(),
//                       local_decay,
//                       temp_[param_id]->gpu_data(),
//                       net_params[param_id]->mutable_gpu_diff());
//      } else {
//        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
//      }
//    }
//#else
//    NO_GPU;
//#endif
//    break;
//  }
//  default:
//    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
//  }
//}
//
//template <typename Dtype>//added 10
//void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
//  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
//  const vector<float>& net_params_lr = this->net_->params_lr();
//  Dtype momentum = this->param_.momentum();
//  Dtype local_rate = rate * net_params_lr[param_id];
//  // Compute the update to history, then copy it to the parameter diff.
//  switch (Caffe::mode()) {
//  case Caffe::CPU: {
//    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
//                    net_params[param_id]->cpu_diff(), momentum,
//                    history_[param_id]->mutable_cpu_data());
//    caffe_copy(net_params[param_id]->count(),
//               history_[param_id]->cpu_data(),
//               net_params[param_id]->mutable_cpu_diff());
//    break;
//  }
//  case Caffe::GPU: {
//#ifndef CPU_ONLY
//    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
//                    net_params[param_id]->gpu_diff(), momentum,
//                    history_[param_id]->mutable_gpu_data());
//    caffe_copy(net_params[param_id]->count(),
//               history_[param_id]->gpu_data(),
//               net_params[param_id]->mutable_gpu_diff());
//#else
//    NO_GPU;
//#endif
//    break;
//  }
//  default:
//    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
//  }
//}
//
//template <typename Dtype>//added 11
//void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
//  switch (this->param_.snapshot_format()) {
//  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
//    SnapshotSolverStateToBinaryProto(model_filename);
//    break;
//  case caffe::SolverParameter_SnapshotFormat_HDF5:
//    SnapshotSolverStateToHDF5(model_filename);
//    break;
//  default:
//    LOG(FATAL) << "Unsupported snapshot format.";
//  }
//}
//
//template <typename Dtype>//added 12
//void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(
//  const string& model_filename) {
//  SolverState state;
//  state.set_iter(this->iter_);
//  state.set_learned_net(model_filename);
//  state.set_current_step(this->current_step_);
//  state.clear_history();
//  for (int i = 0; i < history_.size(); ++i) {
//    // Add history
//    BlobProto* history_blob = state.add_history();
//    history_[i]->ToProto(history_blob);
//  }
//  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
//  LOG(INFO)
//      << "Snapshotting solver state to binary proto file" << snapshot_filename;
//  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
//}
//
//template <typename Dtype>//added 13
//void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(
//  const string& model_filename) {
//  string snapshot_filename =
//    Solver<Dtype>::SnapshotFilename(".solverstate.h5");
//  LOG(INFO) << "Snapshotting solver state to HDF5 file " << snapshot_filename;
//  hid_t file_hid = H5Fcreate(snapshot_filename.c_str(), H5F_ACC_TRUNC,
//                             H5P_DEFAULT, H5P_DEFAULT);
//  CHECK_GE(file_hid, 0)
//      << "Couldn't open " << snapshot_filename << " to save solver state.";
//  hdf5_save_int(file_hid, "iter", this->iter_);
//  hdf5_save_string(file_hid, "learned_net", model_filename);
//  hdf5_save_int(file_hid, "current_step", this->current_step_);
//  hid_t history_hid = H5Gcreate2(file_hid, "history", H5P_DEFAULT, H5P_DEFAULT,
//                                 H5P_DEFAULT);
//  CHECK_GE(history_hid, 0)
//      << "Error saving solver state to " << snapshot_filename << ".";
//  for (int i = 0; i < history_.size(); ++i) {
//    ostringstream oss;
//    oss << i;
//    hdf5_save_nd_dataset<Dtype>(history_hid, oss.str(), *history_[i]);
//  }
//  H5Gclose(history_hid);
//  H5Fclose(file_hid);
//}
//
//template <typename Dtype>//added 14
//void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(
//  const string& state_file) {
//  SolverState state;
//  ReadProtoFromBinaryFile(state_file, &state);
//  this->iter_ = state.iter();
//  if (state.has_learned_net()) {
//    NetParameter net_param;
//    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
//    this->net_->CopyTrainedLayersFrom(net_param);
//  }
//  this->current_step_ = state.current_step();
//  CHECK_EQ(state.history_size(), history_.size())
//      << "Incorrect length of history blobs.";
//  LOG(INFO) << "SGDSolver: restoring history";
//  for (int i = 0; i < history_.size(); ++i) {
//    history_[i]->FromProto(state.history(i));
//  }
//}
//
//template <typename Dtype>//added 15
//void SGDSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
//  hid_t file_hid = H5Fopen(state_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
//  CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_file;
//  this->iter_ = hdf5_load_int(file_hid, "iter");
//  if (H5LTfind_dataset(file_hid, "learned_net")) {
//    string learned_net = hdf5_load_string(file_hid, "learned_net");
//    this->net_->CopyTrainedLayersFrom(learned_net);
//  }
//  this->current_step_ = hdf5_load_int(file_hid, "current_step");
//  hid_t history_hid = H5Gopen2(file_hid, "history", H5P_DEFAULT);
//  CHECK_GE(history_hid, 0) << "Error reading history from " << state_file;
//  int state_history_size = hdf5_get_num_links(history_hid);
//  CHECK_EQ(state_history_size, history_.size())
//      << "Incorrect length of history blobs.";
//  for (int i = 0; i < history_.size(); ++i) {
//    ostringstream oss;
//    oss << i;
//    hdf5_load_nd_dataset<Dtype>(history_hid, oss.str().c_str(), 0,
//                                kMaxBlobAxes, history_[i].get());
//  }
//  H5Gclose(history_hid);
//  H5Fclose(file_hid);
//}
//
//template <typename Dtype>//added 16
//void NesterovSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
//  CHECK(Caffe::root_solver());
//  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
//  const vector<float>& net_params_lr = this->net_->params_lr();
//  Dtype momentum = this->param_.momentum();
//  Dtype local_rate = rate * net_params_lr[param_id];
//  switch (Caffe::mode()) {
//  case Caffe::CPU: {
//    // save history momentum for stepping back
//    caffe_copy(net_params[param_id]->count(),
//               this->history_[param_id]->cpu_data(),
//               this->update_[param_id]->mutable_cpu_data());
//
//    // update history
//    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
//                    net_params[param_id]->cpu_diff(), momentum,
//                    this->history_[param_id]->mutable_cpu_data());
//
//    // compute update: step back then over step
//    caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
//                    this->history_[param_id]->cpu_data(), -momentum,
//                    this->update_[param_id]->mutable_cpu_data());
//
//    // copy
//    caffe_copy(net_params[param_id]->count(),
//               this->update_[param_id]->cpu_data(),
//               net_params[param_id]->mutable_cpu_diff());
//    break;
//  }
//  case Caffe::GPU: {
//#ifndef CPU_ONLY
//    // save history momentum for stepping back
//    caffe_copy(net_params[param_id]->count(),
//               this->history_[param_id]->gpu_data(),
//               this->update_[param_id]->mutable_gpu_data());
//
//    // update history
//    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
//                    net_params[param_id]->gpu_diff(), momentum,
//                    this->history_[param_id]->mutable_gpu_data());
//
//    // compute update: step back then over step
//    caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
//                    this->history_[param_id]->gpu_data(), -momentum,
//                    this->update_[param_id]->mutable_gpu_data());
//
//    // copy
//    caffe_copy(net_params[param_id]->count(),
//               this->update_[param_id]->gpu_data(),
//               net_params[param_id]->mutable_gpu_diff());
//#else
//    NO_GPU;
//#endif
//    break;
//  }
//  default:
//    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
//  }
//}
//
//template <typename Dtype>//added 17
//void AdaGradSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
//  CHECK(Caffe::root_solver());
//  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
//  const vector<float>& net_params_lr = this->net_->params_lr();
//  Dtype delta = this->param_.delta();
//  Dtype local_rate = rate * net_params_lr[param_id];
//  switch (Caffe::mode()) {
//  case Caffe::CPU: {
//    // compute square of gradient in update
//    caffe_powx(net_params[param_id]->count(),
//               net_params[param_id]->cpu_diff(), Dtype(2),
//               this->update_[param_id]->mutable_cpu_data());
//
//    // update history
//    caffe_add(net_params[param_id]->count(),
//              this->update_[param_id]->cpu_data(),
//              this->history_[param_id]->cpu_data(),
//              this->history_[param_id]->mutable_cpu_data());
//
//    // prepare update
//    caffe_powx(net_params[param_id]->count(),
//               this->history_[param_id]->cpu_data(), Dtype(0.5),
//               this->update_[param_id]->mutable_cpu_data());
//
//    caffe_add_scalar(net_params[param_id]->count(),
//                     delta, this->update_[param_id]->mutable_cpu_data());
//
//    caffe_div(net_params[param_id]->count(),
//              net_params[param_id]->cpu_diff(),
//              this->update_[param_id]->cpu_data(),
//              this->update_[param_id]->mutable_cpu_data());
//
//    // scale and copy
//    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
//                    this->update_[param_id]->cpu_data(), Dtype(0),
//                    net_params[param_id]->mutable_cpu_diff());
//    break;
//  }
//  case Caffe::GPU: {
//#ifndef CPU_ONLY
//    // compute square of gradient in update
//    caffe_gpu_powx(net_params[param_id]->count(),
//                   net_params[param_id]->gpu_diff(), Dtype(2),
//                   this->update_[param_id]->mutable_gpu_data());
//
//    // update history
//    caffe_gpu_add(net_params[param_id]->count(),
//                  this->update_[param_id]->gpu_data(),
//                  this->history_[param_id]->gpu_data(),
//                  this->history_[param_id]->mutable_gpu_data());
//
//    // prepare update
//    caffe_gpu_powx(net_params[param_id]->count(),
//                   this->history_[param_id]->gpu_data(), Dtype(0.5),
//                   this->update_[param_id]->mutable_gpu_data());
//
//    caffe_gpu_add_scalar(net_params[param_id]->count(),
//                         delta, this->update_[param_id]->mutable_gpu_data());
//
//    caffe_gpu_div(net_params[param_id]->count(),
//                  net_params[param_id]->gpu_diff(),
//                  this->update_[param_id]->gpu_data(),
//                  this->update_[param_id]->mutable_gpu_data());
//
//    // scale and copy
//    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
//                    this->update_[param_id]->gpu_data(), Dtype(0),
//                    net_params[param_id]->mutable_gpu_diff());
//#else
//    NO_GPU;
//#endif
//    break;
//  }
//  default:
//    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
//  }
//}
//
//template <typename Dtype>//added 18
//void RMSPropSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
//  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
//  const vector<float>& net_params_lr = this->net_->params_lr();
//
//  // get the learning rate
//  Dtype delta = this->param_.delta();
//  Dtype rms_decay = this->param_.rms_decay();
//  Dtype local_rate = rate * net_params_lr[param_id];
//
//  switch (Caffe::mode()) {
//  case Caffe::CPU:
//    // compute square of gradient in update
//    caffe_powx(net_params[param_id]->count(),
//               net_params[param_id]->cpu_diff(), Dtype(2),
//               this->update_[param_id]->mutable_cpu_data());
//
//    // update history
//    caffe_cpu_axpby(net_params[param_id] -> count(),
//                    Dtype(1 - rms_decay), this->update_[param_id]->cpu_data(),
//                    rms_decay, this->history_[param_id]-> mutable_cpu_data());
//
//    // prepare update
//    caffe_powx(net_params[param_id]->count(),
//               this->history_[param_id]->cpu_data(), Dtype(0.5),
//               this->update_[param_id]->mutable_cpu_data());
//
//    caffe_add_scalar(net_params[param_id]->count(),
//                     delta, this->update_[param_id]->mutable_cpu_data());
//
//    caffe_div(net_params[param_id]->count(),
//              net_params[param_id]->cpu_diff(), this->update_[param_id]->cpu_data(),
//              this->update_[param_id]->mutable_cpu_data());
//
//    // scale and copy
//    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
//                    this->update_[param_id]->cpu_data(), Dtype(0),
//                    net_params[param_id]->mutable_cpu_diff());
//    break;
//  case Caffe::GPU:
//#ifndef CPU_ONLY
//    // compute square of gradient in update
//    caffe_gpu_powx(net_params[param_id]->count(),
//                   net_params[param_id]->gpu_diff(), Dtype(2),
//                   this->update_[param_id]->mutable_gpu_data());
//
//    // update history
//    caffe_gpu_axpby(net_params[param_id] -> count(),
//                    Dtype(1 - rms_decay), this->update_[param_id]->gpu_data(),
//                    rms_decay, this->history_[param_id]-> mutable_gpu_data());
//
//    // prepare update
//    caffe_gpu_powx(net_params[param_id]->count(),
//                   this->history_[param_id]->gpu_data(), Dtype(0.5),
//                   this->update_[param_id]->mutable_gpu_data());
//
//    caffe_gpu_add_scalar(net_params[param_id]->count(),
//                         delta, this->update_[param_id]->mutable_gpu_data());
//
//    caffe_gpu_div(net_params[param_id]->count(),
//                  net_params[param_id]->gpu_diff(), this->update_[param_id]->gpu_data(),
//                  this->update_[param_id]->mutable_gpu_data());
//
//    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
//                    this->update_[param_id]->gpu_data(), Dtype(0),
//                    net_params[param_id]->mutable_gpu_diff());
//#else
//    NO_GPU;
//#endif
//    break;
//  default:
//    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
//  }
//}
//
//template <typename Dtype>//added 19
//void AdaDeltaSolver<Dtype>::AdaDeltaPreSolve() {
//  // Add the extra history entries for AdaDelta after those from
//  // SGDSolver::PreSolve
//  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
//  for (int i = 0; i < net_params.size(); ++i) {
//    const vector<int>& shape = net_params[i]->shape();
//    this->history_.push_back(
//      shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
//  }
//}
//
//template <typename Dtype>//added 20
//void AdaDeltaSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
//  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
//  const vector<float>& net_params_lr = this->net_->params_lr();
//  Dtype delta = this->param_.delta();
//  Dtype momentum = this->param_.momentum();
//  Dtype local_rate = rate * net_params_lr[param_id];
//  size_t update_history_offset = net_params.size();
//  switch (Caffe::mode()) {
//  case Caffe::CPU: {
//    // compute square of gradient in update
//    caffe_powx(net_params[param_id]->count(),
//               net_params[param_id]->cpu_diff(), Dtype(2),
//               this->update_[param_id]->mutable_cpu_data());
//
//    // update history of gradients
//    caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
//                    this->update_[param_id]->cpu_data(), momentum,
//                    this->history_[param_id]->mutable_cpu_data());
//
//    // add delta to history to guard against dividing by zero later
//    caffe_set(net_params[param_id]->count(), delta,
//              this->temp_[param_id]->mutable_cpu_data());
//
//    caffe_add(net_params[param_id]->count(),
//              this->temp_[param_id]->cpu_data(),
//              this->history_[update_history_offset + param_id]->cpu_data(),
//              this->update_[param_id]->mutable_cpu_data());
//
//    caffe_add(net_params[param_id]->count(),
//              this->temp_[param_id]->cpu_data(),
//              this->history_[param_id]->cpu_data(),
//              this->temp_[param_id]->mutable_cpu_data());
//
//    // divide history of updates by history of gradients
//    caffe_div(net_params[param_id]->count(),
//              this->update_[param_id]->cpu_data(),
//              this->temp_[param_id]->cpu_data(),
//              this->update_[param_id]->mutable_cpu_data());
//
//    // jointly compute the RMS of both for update and gradient history
//    caffe_powx(net_params[param_id]->count(),
//               this->update_[param_id]->cpu_data(), Dtype(0.5),
//               this->update_[param_id]->mutable_cpu_data());
//
//    // compute the update
//    caffe_mul(net_params[param_id]->count(),
//              net_params[param_id]->cpu_diff(),
//              this->update_[param_id]->cpu_data(),
//              net_params[param_id]->mutable_cpu_diff());
//
//    // compute square of update
//    caffe_powx(net_params[param_id]->count(),
//               net_params[param_id]->cpu_diff(), Dtype(2),
//               this->update_[param_id]->mutable_cpu_data());
//
//    // update history of updates
//    caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
//                    this->update_[param_id]->cpu_data(), momentum,
//                    this->history_[update_history_offset + param_id]->mutable_cpu_data());
//
//    // apply learning rate
//    caffe_cpu_scale(net_params[param_id]->count(), local_rate,
//                    net_params[param_id]->cpu_diff(),
//                    net_params[param_id]->mutable_cpu_diff());
//    break;
//  }
//  case Caffe::GPU: {
//#ifndef CPU_ONLY
//    // compute square of gradient in update
//    caffe_gpu_powx(net_params[param_id]->count(),
//                   net_params[param_id]->gpu_diff(), Dtype(2),
//                   this->update_[param_id]->mutable_gpu_data());
//
//    // update history of gradients
//    caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
//                    this->update_[param_id]->gpu_data(), momentum,
//                    this->history_[param_id]->mutable_gpu_data());
//
//    // add delta to history to guard against dividing by zero later
//    caffe_gpu_set(net_params[param_id]->count(), delta,
//                  this->temp_[param_id]->mutable_gpu_data());
//
//    caffe_gpu_add(net_params[param_id]->count(),
//                  this->temp_[param_id]->gpu_data(),
//                  this->history_[update_history_offset + param_id]->gpu_data(),
//                  this->update_[param_id]->mutable_gpu_data());
//
//    caffe_gpu_add(net_params[param_id]->count(),
//                  this->temp_[param_id]->gpu_data(),
//                  this->history_[param_id]->gpu_data(),
//                  this->temp_[param_id]->mutable_gpu_data());
//
//    // divide history of updates by history of gradients
//    caffe_gpu_div(net_params[param_id]->count(),
//                  this->update_[param_id]->gpu_data(),
//                  this->temp_[param_id]->gpu_data(),
//                  this->update_[param_id]->mutable_gpu_data());
//
//    // jointly compute the RMS of both for update and gradient history
//    caffe_gpu_powx(net_params[param_id]->count(),
//                   this->update_[param_id]->gpu_data(), Dtype(0.5),
//                   this->update_[param_id]->mutable_gpu_data());
//
//    // compute the update and copy to net_diff
//    caffe_gpu_mul(net_params[param_id]->count(),
//                  net_params[param_id]->gpu_diff(),
//                  this->update_[param_id]->gpu_data(),
//                  net_params[param_id]->mutable_gpu_diff());
//
//    // compute square of update
//    caffe_gpu_powx(net_params[param_id]->count(),
//                   net_params[param_id]->gpu_diff(), Dtype(2),
//                   this->update_[param_id]->mutable_gpu_data());
//
//    // update history of updates
//    caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
//                    this->update_[param_id]->gpu_data(), momentum,
//                    this->history_[update_history_offset + param_id]->mutable_gpu_data());
//
//    // apply learning rate
//    caffe_gpu_scale(net_params[param_id]->count(), local_rate,
//                    net_params[param_id]->gpu_diff(),
//                    net_params[param_id]->mutable_gpu_diff());
//#else
//    NO_GPU;
//#endif
//    break;
//  }
//  default:
//    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
//  }
//}
//
//template <typename Dtype>//added 21
//void AdamSolver<Dtype>::AdamPreSolve() {
//  // Add the extra history entries for Adam after those from
//  // SGDSolver::PreSolve
//  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
//  for (int i = 0; i < net_params.size(); ++i) {
//    const vector<int>& shape = net_params[i]->shape();
//    this->history_.push_back(
//      shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
//  }
//}
//
//template <typename Dtype>//added 22
//void AdamSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
//  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
//  const vector<float>& net_params_lr = this->net_->params_lr();
//  Dtype local_rate = rate * net_params_lr[param_id];
//  const Dtype beta1 = this->param_.momentum();
//  const Dtype beta2 = this->param_.momentum2();
//
//  // we create aliases for convenience
//  size_t update_history_offset = net_params.size();
//  Blob<Dtype>* val_m = this->history_[param_id].get();
//  Blob<Dtype>* val_v = this->history_[param_id + update_history_offset].get();
//  Blob<Dtype>* val_t = this->temp_[param_id].get();
//
//  const int t = this->iter_  + 1;
//  const Dtype correction = std::sqrt(Dtype(1) - pow(beta2, t)) /
//                           (Dtype(1.) - pow(beta1, t));
//  const int N = net_params[param_id]->count();
//  const Dtype eps_hat = this->param_.delta();
//
//  switch (Caffe::mode()) {
//  case Caffe::CPU: {
//    // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
//    caffe_cpu_axpby(N, Dtype(1) - beta1,
//                    net_params[param_id]->cpu_diff(), beta1,
//                    val_m->mutable_cpu_data());
//
//    // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
//    caffe_mul(N,
//              net_params[param_id]->cpu_diff(),
//              net_params[param_id]->cpu_diff(),
//              val_t->mutable_cpu_data());
//    caffe_cpu_axpby(N, Dtype(1) - beta2,
//                    val_t->cpu_data(), beta2,
//                    val_v->mutable_cpu_data());
//
//    // set update
//    caffe_powx(N,
//               val_v->cpu_data(), Dtype(0.5),
//               val_t->mutable_cpu_data());
//    caffe_add_scalar(N, eps_hat, val_t->mutable_cpu_data());
//    caffe_div(N,
//              val_m->cpu_data(),
//              val_t->cpu_data(),
//              val_t->mutable_cpu_data());
//
//    caffe_cpu_scale(N, local_rate * correction,
//                    val_t->cpu_data(),
//                    net_params[param_id]->mutable_cpu_diff());
//    break;
//  }
//  case Caffe::GPU: {
//#ifndef CPU_ONLY
//    // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
//    caffe_gpu_axpby(N, Dtype(1) - beta1,
//                    net_params[param_id]->gpu_diff(), beta1,
//                    val_m->mutable_gpu_data());
//
//    // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
//    caffe_gpu_mul(N,
//                  net_params[param_id]->gpu_diff(),
//                  net_params[param_id]->gpu_diff(),
//                  val_t->mutable_gpu_data());
//    caffe_gpu_axpby(N, Dtype(1) - beta2,
//                    val_t->gpu_data(), beta2,
//                    val_v->mutable_gpu_data());
//
//    // set update
//    caffe_gpu_powx(N,
//                   val_v->gpu_data(), Dtype(0.5),
//                   val_t->mutable_gpu_data());
//    caffe_gpu_add_scalar(N, eps_hat,
//                         val_t->mutable_gpu_data());
//    caffe_gpu_div(N,
//                  val_m->gpu_data(),
//                  val_t->gpu_data(),
//                  val_t->mutable_gpu_data());
//
//    caffe_gpu_scale(N, local_rate * correction,
//                    val_t->gpu_data(),
//                    net_params[param_id]->mutable_gpu_diff());
//#else
//    NO_GPU;
//#endif
//    break;
//  }
//  default:
//    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
//  }
//}

INSTANTIATE_CLASS(Solver);
//INSTANTIATE_CLASS(SGDSolver);
//INSTANTIATE_CLASS(NesterovSolver);
//INSTANTIATE_CLASS(AdaGradSolver);
//INSTANTIATE_CLASS(RMSPropSolver);
//INSTANTIATE_CLASS(AdaDeltaSolver);
//INSTANTIATE_CLASS(AdamSolver);

}  // namespace caffe
