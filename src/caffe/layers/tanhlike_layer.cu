#include <cmath>
#include <vector>

#include "caffe/neuron_layers.hpp"

namespace caffe {

template<typename Dtype>
__device__  inline Dtype tanhlike_gpu(Dtype x, float beta) {
  return 2. / (1. + exp(-beta * x)) - 1.;
}

template <typename Dtype>
__global__ void TanhlikeForward(const int n, const Dtype* in, Dtype* out, float beta) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = tanhlike_gpu(in[index], beta);
  }
}

template <typename Dtype>
void TanhlikeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  float tanh_beta = this->layer_param_.tanhlike_param().tanh_beta();
  // NOLINT_NEXT_LINE(whitespace/operators)
  TanhlikeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, tanh_beta);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void TanhlikeBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff, float beta) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype tanh_x = out_data[index];
    out_diff[index] = 0.5 * beta * in_diff[index] * (1. + tanh_x) * (1. - tanh_x);
  }
}

template <typename Dtype>
void TanhlikeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    float tanh_beta = this->layer_param_.tanhlike_param().tanh_beta();
    // NOLINT_NEXT_LINE(whitespace/operators)
    TanhlikeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, bottom_diff, tanh_beta);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TanhlikeLayer);


}  // namespace caffe
