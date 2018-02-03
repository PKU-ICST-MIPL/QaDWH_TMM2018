//Created by Ye Liu (E-mail: jourkliu@163.com) from Sun Yat-sen University @ 2014-12-26

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void MultiLabelSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Sigmoid");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  LOG(INFO) << "MulitlabelSoftMax: LayerSetup: Before lastline";
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void MultiLabelSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  //softmax_layer_->Reshape(softmax_bottom_vec_, &softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void MultiLabelSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  //softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; j++) {
	  if(label[i*dim + j] > 0){
		loss += -log(max(prob_data[i * dim + j], Dtype(FLT_MIN)));
	  }
    }
  }
  //(*top)[0]->mutable_cpu_data()[0] = loss / num / dim;
  top[0]->mutable_cpu_data()[0] = loss / num;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void MultiLabelSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    //LOG(INFO) << "All Sample Num: " << num << " Sample Dim:" << dim;
	//LOG(INFO) << "FLT_MIN = " << FLT_MIN << " kLOG_THRESHOLD" << kLOG_THRESHOLD;
    Dtype loss = 0;
    for (int i = 0; i < num; ++i) {
      //LOG(INFO) << "num: " << i;
      for (int j = 0; j < dim; ++j) {
        //LOG(INFO) << "label: " << label[i*dim + j] << " bottom_diff:" << bottom_diff[i * dim + j];
	    if(label[i*dim + j] > 0){
          bottom_diff[i * dim + j] -= 1.0;
		  loss += -log(max(prob_data[i * dim + j], Dtype(FLT_MIN)));
		}
      }
    }
    //LOG(INFO) << "loss = " << loss / num;
    // Scale gradient
    //const Dtype loss_weight = top[0]->cpu_diff()[0];
    //LOG(INFO) << "loss_weight = " << loss_weight;
	//loss_weight equal to 1
    //caffe_scal(prob_.count(), loss_weight / num / dim, bottom_diff);
	caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(MultiLabelSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(MultiLabelSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(MultiLabelSoftmaxWithLoss);

}  // namespace caffe
