#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void ElementWiseProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / bottom[0]->num();

  //caffe_gpu_mul( bottom[0]->count(), bottom[0]->gpu_data(), weight, top_data);

  for ( int i = 0; i < num ; i++ )
  {
    const Dtype* data = bottom_data + i* dim;
    caffe_gpu_mul( dim, data, weight, top_data + i*dim );
  }
}


template<typename Dtype>
void ElementWiseProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
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

    caffe_gpu_axpy( dim, Dtype(1), intermediate_result.gpu_data(), this->blobs_[0]->mutable_gpu_diff() );

  }


  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    for ( int i = 0; i <= num ; i++ )
    {
      const Dtype* diff_data = top_diff + i* dim;
      Dtype* diff_data_b = bottom_diff + i* dim;
      caffe_gpu_mul( dim, diff_data, weight, diff_data_b );
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ElementWiseProductLayer);

}  // namespace caffe
