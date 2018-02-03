#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // number of triplet in a batch
  int num_triplets = this->layer_param_.triplet_loss_param().num_triplets();
  float margin = this->layer_param_.triplet_loss_param().margin();
  // dimension of each descriptor
  int dim = bottom[0]->count()/bottom[0]->num();
  CHECK_EQ(bottom[0]->channels(), dim);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  //printf("%d %d\n",bottom[0]->count(),dim);
  //bottom[1] label
  //CHECK_EQ(bottom[1]->channels(), 1);
  //CHECK_EQ(bottom[1]->height(), 1);
  //CHECK_EQ(bottom[1]->width(), 1);
  
  // In each set, we have:
  // the descriptor of reference sample, closest sample, and negative samples
  // number of sets in the whole batch
  int num_set = bottom[0]->num()/num_triplets;	// we only use reference sample, closest sample, and negative sample
  printf("num_set_layer:%d\n",num_set);
  LOG(INFO)<<"triplet tuple num: "<<num_set<<", hash code bit num: "<<dim<<", margin: "<<margin;
  dist_sq_.Reshape(num_set, 1, 1, 1);
  diff_pos.Reshape(num_set, dim, 1, 1);
  dist_sq_pos.Reshape(num_set, 1, 1, 1);
  diff_neg.Reshape(num_set, dim, 1, 1);
  dist_sq_neg.Reshape(num_set, 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  /*
  for(int i = 0;i < bottom[0]->count();i ++)
  {
	  Dtype bottom_tmp = Dtype(0.0);
	  bottom_tmp = bottom[0]->cpu_data()[i];
          if( bottom_tmp < Dtype(0.52) && bottom_tmp > Dtype(0.48))
	    printf("bottom_tmp:%f\n",bottom_tmp);
	  if(bottom_tmp > Dtype(0.52))
		  bottom_tmp = Dtype(1.0);
          else if(bottom_tmp < Dtype(0.48))
		  bottom_tmp = Dtype(0.0);
	  bottom[0]->mutable_cpu_data()[i] = bottom_tmp;
  }
  */
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  //Dtype losstype = this->layer_param_.triplet_loss_param().losstype();
  int num_triplets = this->layer_param_.triplet_loss_param().num_triplets();
  CHECK_EQ(bottom[0]->num()%num_triplets,0);
  //CHECK_EQ(bottom[0]->num()%(2 + num_triplets), 0);
  Dtype loss(0.0);
  int dim = bottom[0]->count()/bottom[0]->num();
  //int num_set = bottom[0]->num()/(2 + num_triplets);
  int num_set = bottom[0]->num()/num_triplets;
  for (int i = 0; i < num_set; ++i) {
	
	// added by zjj
	caffe_sub(
		dim,
		bottom[0]->cpu_data() + num_triplets*i*dim, //reference
		bottom[0]->cpu_data() + (num_triplets*i + 1)*dim, //positive
		diff_pos.mutable_cpu_data() + i*dim);
	dist_sq_pos.mutable_cpu_data()[i] = caffe_cpu_dot(dim,
		diff_pos.cpu_data() + i*dim,
		diff_pos.cpu_data() + i*dim);
	dist_sq_.mutable_cpu_data()[i] = dist_sq_pos.cpu_data()[i];
	caffe_sub(
		dim,
		bottom[0]->cpu_data() + num_triplets*i*dim, //reference
		bottom[0]->cpu_data() + (num_triplets*i + 2)*dim, //negative
		diff_neg.mutable_cpu_data() + i*dim);
	dist_sq_neg.mutable_cpu_data()[i] = caffe_cpu_dot(dim,
		diff_neg.cpu_data() + i*dim,
		diff_neg.cpu_data() + i*dim);
	dist_sq_.mutable_cpu_data()[i] -= dist_sq_neg.cpu_data()[i];
	loss += std::max(margin + dist_sq_.cpu_data()[i],Dtype(0.0));
	
  }
  loss = loss / static_cast<Dtype>(num_set) / Dtype(2);
  //printf("loss %f\n",loss);
  printf("dist_sq_ [0]:%f [%d]%f\tdim:%d\tloss:%f\n", dist_sq_.cpu_data()[0],num_set - 1,dist_sq_.cpu_data()[num_set-1],dim,loss);
  top[0]->mutable_cpu_data()[0] = loss;
  
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  //Dtype losstype = this->layer_param_.triplet_loss_param().losstype();
  int num_triplets = this->layer_param_.triplet_loss_param().num_triplets();
  int dim = bottom[0]->count()/bottom[0]->num();
  //int num_set = bottom[0]->num()/(2 + num_triplets);
  CHECK_EQ(bottom[0]->num()%num_triplets, 0);
  int num_set = bottom[0]->num()/num_triplets;
  printf("backward dist_sq_ %f %f\n", dist_sq_.cpu_data()[0],dist_sq_.cpu_data()[num_set-1]);
  const Dtype* label = bottom[1]->cpu_data();
  int num_valid = 0;
  for(int i = 0;i < 3;i ++) {
	if (propagate_down[0]) {
	  const Dtype sign = 1;
	  const Dtype alpha = sign * top[0]->cpu_diff()[0] / 
		  static_cast<Dtype>(num_set);
	  for (int j = 0;j < num_set;j ++) {
	     //CHECK_EQ(label[j*num_triplets],label[j*num_triplets+1]);
	     //CHECK(label[j*num_triplets+1]!=label[j*num_triplets+2]);
		Dtype* bout = bottom[0]->mutable_cpu_diff();
		if (dist_sq_.cpu_data()[j] + margin > 0) {
			// BP for feat1(extracted from reference)
			if(i == 0) {
				caffe_cpu_axpby(
					dim,
					alpha,
					diff_pos.cpu_data() + (j*dim),
					Dtype(0.0),
					bout + (num_triplets*j + i)*dim);
				caffe_cpu_axpby(
					dim,
					-alpha,
					diff_neg.cpu_data() + (j*dim),
					Dtype(1.0),
					bout + (num_triplets*j + i)*dim);
			}
			// BP for feat2(extracted from positive)
			if(i == 1) {
				caffe_cpu_axpby(
					dim,
					-alpha,
					diff_pos.cpu_data() + (j*dim),
					Dtype(0.0),
					bout + (num_triplets*j + i)*dim);
			}
			// BP for feat3(extracted from negative)
			if(i == 2) {
			   num_valid++;
				caffe_cpu_axpby(
					dim,
					alpha,
					diff_neg.cpu_data() + (j*dim),
					Dtype(0.0),
					bout + (num_triplets*j + i)*dim);
			}
		}
		else {	// 0
			caffe_set(dim,Dtype(0.0),bout + (num_triplets*j + i)*dim);
			// caffe_sub(
					// dim,
					// bout + (num_triplets*j + i)*dim,
					// bout + (num_triplets*j + i)*dim,
					// bout + (num_triplets*j + i)*dim);
		}
	  }
	}
  }
  if(bottom.size()==3){
  const Dtype* prob_data = bottom[2]->cpu_data();
  int num = bottom[2]->num();
  int clsnum = bottom[2]->count()/num;
  int pred_label = 0;
  int true_cnt = 0;
  for(int i=0;i<num;i++){
     pred_label = 0;
     for(int j=0;j<clsnum;j++){
        if(prob_data[i*clsnum+j]>prob_data[i*clsnum+pred_label]){
           pred_label = j;
        }
     }
     if(static_cast<Dtype>(pred_label)==label[i]){
        true_cnt++;
     }
  }
  printf("num_set: %d, num_valid: %d;     batch_size: %d, clsnum: %d, true_cnt: %d, true_rate: %.3f ...\n", num_set, num_valid, num, clsnum, true_cnt,1.0*true_cnt/num);
  }
  //printf("alpha%f\n",top[0]->cpu_diff()[0] / static_cast<Dtype>(num_set));
  printf("diff:pos-neg:%f\tpos:%f\tneg:%f\n",diff_pos.cpu_data()[0]*2-diff_neg.cpu_data()[0]*2,diff_pos.cpu_data()[0]*2,diff_neg.cpu_data()[0]*2);
  printf("bout_[0]:%f[1]:%f[2]:%f\n",bottom[0]->mutable_cpu_diff()[0],bottom[0]->mutable_cpu_diff()[1],bottom[0]->mutable_cpu_diff()[2]);
  printf("num_set: %d, num_valid: %d ...\n", num_set, num_valid);
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe
