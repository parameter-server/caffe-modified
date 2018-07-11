#include <vector>
#include <omp.h>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  int threads = omp_get_max_threads();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    #pragma omp parallel for num_threads(threads)
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  int threads = omp_get_max_threads();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      this->clear_bias_buffer();
    }
    this->clear_weight_buffer();
    
    #pragma omp parallel for num_threads(threads)
    for (int n = 0; n < this->num_; ++n) {
	    if (this->bias_term_ && this->param_propagate_down_[1]) {
		    this->backward_cpu_bias_omp(top_diff + top[i]->offset(n));
	    }
      if (this->param_propagate_down_[0]) {
        this->weight_cpu_gemm_omp(bottom_data + bottom[i]->offset(n),
          top_diff + top[i]->offset(n));
      }
      if (propagate_down[i]) {
        this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
          bottom_diff + bottom[i]->offset(n));
      }
    }

	  if (this->bias_term_ && this->param_propagate_down_[1]) {
	    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
	    this->sum_bias_buffer(bias_diff);
  	}
	  if (this->param_propagate_down_[0]) {
	    this->sum_weight_buffer(weight_diff);
  	}
  }
}


#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
