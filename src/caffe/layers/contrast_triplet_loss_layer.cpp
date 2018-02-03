#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ContrastTripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // number of items in a tuple
  int tuple_dim = this->layer_param_.contrast_triplet_loss_param().tuple_dim();
  
  CHECK_EQ(bottom[0]->num()%tuple_dim, 0);
  
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  
  int num_set = bottom[0]->num()/tuple_dim;	//we sample one positive and one negative for each reference 
  int dim = bottom[0]->count()/bottom[0]->num();
  Dtype margin = this->layer_param_.contrast_triplet_loss_param().margin();
  LOG(INFO)<<"tuple_dim: "<<tuple_dim<<", tuple_num: "<<num_set<<", hash bit num: "<<dim<<", margin: "<<margin;
  
  
  diff_pos_.Reshape(num_set, dim, 1, 1);
  diff_neg_.Reshape(num_set, dim, 1, 1);
  dist_sq_pos_.Reshape(num_set, 1, 1, 1);
  dist_sq_neg_.Reshape(num_set, 1, 1, 1);
  
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void ContrastTripletLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  caffe_set(diff_pos_.count(), Dtype(0.0), diff_pos_.mutable_cpu_data());
  caffe_set(diff_neg_.count(), Dtype(0.0), diff_neg_.mutable_cpu_data());
  caffe_set(dist_sq_pos_.count(), Dtype(0.0), dist_sq_pos_.mutable_cpu_data());
  caffe_set(dist_sq_neg_.count(), Dtype(0.0), dist_sq_neg_.mutable_cpu_data());


  Dtype margin = this->layer_param_.contrast_triplet_loss_param().margin();
  
  int tuple_dim = this->layer_param_.contrast_triplet_loss_param().tuple_dim();
  int num_set = bottom[0]->num()/tuple_dim;
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();

  Dtype loss(0.0);
  int dim = bottom[0]->count()/bottom[0]->num();
  int anc_idx, pos_idx, neg_idx;
  for (int i = 0; i < num_set; ++i) {
  	anc_idx = i*tuple_dim;
  	pos_idx = anc_idx+1;
  	neg_idx = anc_idx+2;
  	CHECK_EQ(bottom_label[anc_idx],bottom_label[pos_idx]);
  	CHECK(bottom_label[anc_idx]!=bottom_label[neg_idx]);
	
	caffe_sub(dim,
		bottom_data + anc_idx*dim, //reference
		bottom_data + pos_idx*dim, //positive
		diff_pos_.mutable_cpu_data() + i*dim);
	dist_sq_pos_.mutable_cpu_data()[i] = caffe_cpu_dot(dim,
		diff_pos_.cpu_data() + i*dim, diff_pos_.cpu_data() + i*dim);
	loss += dist_sq_pos_.cpu_data()[i];
	
	caffe_sub(dim,
		bottom_data + anc_idx*dim, //reference
		bottom_data + neg_idx*dim, //negative
		diff_neg_.mutable_cpu_data() + i*dim);
	dist_sq_neg_.mutable_cpu_data()[i] = caffe_cpu_dot(dim,
		diff_neg_.cpu_data() + i*dim, diff_neg_.cpu_data() + i*dim);
	Dtype dist_d = std::max(static_cast<Dtype>(margin-sqrt(dist_sq_neg_.cpu_data()[i])),Dtype(0.0));
	loss += dist_d*dist_d;
	
  }
  loss = loss / static_cast<Dtype>(num_set) / Dtype(2.0);
  printf("dist_sq_pos_ [0]:%.3f,\t[1]:%.3f,\t[2]:%.3f\n", dist_sq_pos_.cpu_data()[0], dist_sq_pos_.cpu_data()[1], dist_sq_pos_.cpu_data()[2]);
  printf("dist_sq_neg_ [0]:%.3f,\t[1]:%.3f,\t[2]:%.3f. num_set:%d, loss:%f\n", 
  	dist_sq_neg_.cpu_data()[0], dist_sq_neg_.cpu_data()[1], dist_sq_neg_.cpu_data()[2],
  	num_set, loss);
  
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ContrastTripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  //const Dtype* bottom_data = bottom[0]->cpu_data();
  caffe_set(bottom[0]->count(), Dtype(0.0), bottom_diff);

  Dtype margin = this->layer_param_.contrast_triplet_loss_param().margin();
  int tuple_dim = this->layer_param_.contrast_triplet_loss_param().tuple_dim();
  CHECK_EQ(bottom[0]->num()%tuple_dim, 0);
  int dim = bottom[0]->count()/bottom[0]->num();
  int num_set = bottom[0]->num()/tuple_dim;
  int anc_idx, pos_idx, neg_idx;
  int num_valid = 0;
  Dtype alpha = top[0]->cpu_diff()[0] / num_set;
  for(int i=0; i<num_set; i++){
  	anc_idx = i*tuple_dim;
  	pos_idx = anc_idx+1;
  	neg_idx = anc_idx+2;
  	//for pos pair
  	caffe_cpu_axpby(dim, alpha, diff_pos_.cpu_data()+i*dim, Dtype(1.0), bottom_diff + anc_idx*dim);
  	caffe_cpu_axpby(dim, -alpha, diff_pos_.cpu_data()+i*dim, Dtype(1.0), bottom_diff + pos_idx*dim);
  	//for neg pair
  	Dtype dist = sqrt(dist_sq_neg_.cpu_data()[i]);
  	Dtype mdist = margin-dist;
  	Dtype beta = -alpha*mdist / (dist + Dtype(1e-4));
  	if(mdist > Dtype(0.0)){
  	  num_valid++;
  	  caffe_cpu_axpby(dim, beta, diff_neg_.cpu_data()+i*dim, Dtype(1.0), bottom_diff + anc_idx*dim);
  	  caffe_cpu_axpby(dim, -beta, diff_neg_.cpu_data()+i*dim, Dtype(1.0), bottom_diff + neg_idx*dim);
  	}
  } 

  printf("diff:\tpos:%f\t%f\t%f\n\tneg:%f\t%f\t%f, num_valid: %d\n",diff_pos_.cpu_data()[0],diff_pos_.cpu_data()[1],diff_pos_.cpu_data()[2],
  	diff_neg_.cpu_data()[0],diff_neg_.cpu_data()[1],diff_neg_.cpu_data()[2],num_valid);
  printf("bout: [0]:%f\t[1]:%f\t[2]:%f\n",bottom_diff[0],bottom_diff[1],bottom_diff[2]);
}

#ifdef CPU_ONLY
STUB_GPU(ContrastTripletLossLayer);
#endif

INSTANTIATE_CLASS(ContrastTripletLossLayer);
REGISTER_LAYER_CLASS(ContrastTripletLoss);

}  // namespace caffe
