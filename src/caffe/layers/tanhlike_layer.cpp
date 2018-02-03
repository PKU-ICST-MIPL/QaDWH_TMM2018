#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/neuron_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype tanhlike(Dtype x, float beta) {
  return 2. / (1. + exp(-beta * x)) - 1;
}

template <typename Dtype>
void TanhlikeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  float tanh_beta = this->layer_param_.tanhlike_param().tanh_beta();
  for (int i = 0; i < count; ++i) {
    top_data[i] = tanhlike(bottom_data[i], tanh_beta);
  }
}

template <typename Dtype>
void TanhlikeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    float tanh_beta = this->layer_param_.tanhlike_param().tanh_beta();
    for (int i = 0; i < count; ++i) {
      const Dtype tanh_x = top_data[i];
      bottom_diff[i] = 0.5 * tanh_beta * top_diff[i] * (1. + tanh_x) * (1. - tanh_x);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TanhlikeLayer);
#endif

INSTANTIATE_CLASS(TanhlikeLayer);
REGISTER_LAYER_CLASS(Tanhlike);

}  // namespace caffe
