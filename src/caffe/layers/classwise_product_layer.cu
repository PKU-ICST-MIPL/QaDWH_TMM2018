#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void ClassWiseProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
Forward_cpu(bottom, top);
/*
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / bottom[0]->num();
  for ( int i = 0; i < num ; i++ )
  {
    const Dtype* data = bottom_data + i* dim;
    int cur_label = static_cast<int>(bottom_label[i]);
    CHECK_LT(cur_label, C_)<<"error msg: "<<cur_label<<", "<<C_;
    const Dtype* cur_weight = weight + cur_label*dim; 
    caffe_gpu_mul( dim, data, cur_weight, top_data + i*dim );
  }
*/
}


template<typename Dtype>
void ClassWiseProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
Backward_cpu(top, propagate_down, bottom);
/*
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int num = top[0]->num();
  const int dim = top[0]->count() / top[0]->num();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* weight = this->blobs_[0]->gpu_data();



  Blob<Dtype> intermediate_result(1, dim, 1, 1);
  memset(intermediate_result.mutable_cpu_data(), 0, sizeof(Dtype) * dim);


  // Gradient with respect to weight
  for ( int i = 0; i < num ; i++ )
  {

    const Dtype* diff_data = top_diff + i* dim;

    const Dtype* data = bottom_data + i* dim;

    //caffe_mul( dim, diff_data, data, this->blobs_[0]->mutable_gpu_diff() );
    caffe_gpu_mul( dim, diff_data, data, intermediate_result.mutable_gpu_data() );
    int cur_label = static_cast<int>(bottom_label[i]);
    caffe_gpu_axpy( dim, Dtype(1), intermediate_result.gpu_data(), this->blobs_[0]->mutable_gpu_diff()+cur_label*dim);

  }


  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    for ( int i = 0; i <= num ; i++ )
    {
      const Dtype* diff_data = top_diff + i* dim;
      Dtype* diff_data_b = bottom_diff + i* dim;
      int cur_label = static_cast<int>(bottom_label[i]);
      caffe_gpu_mul( dim, diff_data, weight+cur_label*dim, diff_data_b );
    }
  }
*/
}

INSTANTIATE_LAYER_GPU_FUNCS(ClassWiseProductLayer);

}  // namespace caffe
