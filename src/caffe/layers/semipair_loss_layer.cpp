// Initial version
#include <algorithm>
#include <vector>
#include <typeinfo>
#include <float.h>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}
template <typename Dtype>
inline Dtype logExp(Dtype x) {
  if(typeid(x)==typeid(float) && x>Dtype(88.0)){
    return x;
  }
  else if(typeid(x)==typeid(double) && x>Dtype(709.0)){
    return x;
  }
  else{
    return log(1.0+exp(x));
  }
}


template <typename Dtype>
void SemiPairLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  int dim = bottom[0]->count()/bottom[0]->num();
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int fea_index = this->layer_param_.semipair_loss_param().feature_index();
  LOG(INFO)<<"cout: "<<count<<" --- num: "<<num<<" --- dim: "<<dim<<"--- feature dim: "<<bottom[fea_index]->channels()<<"\n";
  CHECK_EQ(bottom[0]->channels(), dim);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->num(),bottom[0]->num());
  CHECK_EQ(bottom[2]->height(),1);
  CHECK_EQ(bottom[2]->width(),1);
  //distance type
  simitype = string("Euclidean");
  simi_s.Reshape(num,num,1,1);
  theta.Reshape(num,num,1,1);
}


template <typename Dtype>
struct SimiNode{
  Dtype simi;
  int index;
  SimiNode(Dtype value,int idx){
    simi = value;
    index = idx;
  }
  friend bool operator<(const SimiNode& sn1,const SimiNode& sn2){
    return sn1.simi>=sn2.simi;
  }
};

template <typename Dtype>
void SemiPairLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype loss(0.0);
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count/num;
  int fea_index = this->layer_param_.semipair_loss_param().feature_index();
  CHECK(fea_index==0||fea_index==2);
  const Dtype* feature = bottom[fea_index]->cpu_data();
  int fea_dim = bottom[fea_index]->channels();
  printf("fea_dim:%d--",fea_dim);

  //reset
  caffe_set(num*num,Dtype(-1.0),simi_s.mutable_cpu_data());
  caffe_set(num*num,Dtype(0.0),theta.mutable_cpu_data());

  const Dtype* bottom_data = bottom[0]->cpu_data(); // bottom_data is n*dim matrix
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* simi_s_data = simi_s.mutable_cpu_data();
  Dtype* theta_data = theta.mutable_cpu_data();
  
  // Calculate S matrix online
  //printf("simi_vec size(top 20): ");
  for(int i = 0; i<num; i++){
    vector<SimiNode<Dtype> > simi_vec;
    Blob<Dtype> diff(fea_dim,1,1,1);
    for(int j=0;j<num;j++){
      CHECK_EQ(simi_s_data[i*num+j],Dtype(-1.0));
      if(bottom_label[i]==Dtype(-1.0)||bottom_label[j]==Dtype(-1.0)){
        caffe_sub(fea_dim,feature+i*fea_dim,feature+j*fea_dim,diff.mutable_cpu_data());
        Dtype dist = caffe_cpu_dot(fea_dim,diff.cpu_data(),diff.cpu_data());
        simi_vec.push_back(SimiNode<Dtype>(-dist,j));
      }
      else{
        if (bottom_label[i] == bottom_label[j]){
          simi_s_data[i*num+j] = Dtype(1.0);
        }
        else{
          simi_s_data[i*num+j] = Dtype(0.0);
        }
      }
    }
    std::sort(simi_vec.begin(),simi_vec.end());
    //if(i<20) printf(" %d",static_cast<int>(simi_vec.size()));
    int value_k = this->layer_param_.semipair_loss_param().knn_k();
    if(value_k<0){
      value_k = static_cast<int>(simi_vec.size())/(-value_k);
    }
    else{
      value_k = std::min(value_k,static_cast<int>(simi_vec.size())/3);
    }
    int s1_max = value_k-1;
    int s0_min = static_cast<int>(simi_vec.size())-value_k;
    //similarity 1
    for(int k=0; k<=s1_max; k++){
      if(k>0){
        CHECK_LE(simi_vec[k].simi,simi_vec[k-1].simi);
      }
      if(simi_s_data[i*num+simi_vec[k].index]<Dtype(0.0)) simi_s_data[i*num+simi_vec[k].index] = Dtype(3.0);
    }
    //similarity -1
    for(int k=s1_max+1;k<s0_min;k++){
      CHECK_LE(simi_vec[k].simi,simi_vec[k-1].simi);
      simi_s_data[i*num+simi_vec[k].index] = Dtype(-1.0);
    }
    //similarity 0
    for(int k = simi_vec.size()-1; k>=s0_min; k--){
      CHECK_LE(simi_vec[k].simi,simi_vec[k-1].simi);
      if(simi_s_data[i*num+simi_vec[k].index]<Dtype(0.0)) simi_s_data[i*num+simi_vec[k].index] = Dtype(2.0);
    }
    //if(i>0 && i%20==0){
      //printf("(%d,%d) ",static_cast<int>(simi_vec.size()),value_k);
    //}
  }
  //printf("\n");
  // post process
  //int num_similar=0,num_disimilar=0,num_ignore=0,num_simi_hyper=0,num_disimi_hyper=0;
  int num_conflict = 0;
  for(int i=0;i<num;i++){
    for(int j=0;j<=i;j++){
      if(simi_s_data[i*num+j]==Dtype(2.0) && simi_s_data[j*num+i]==Dtype(3.0)){
        simi_s_data[i*num+j] = Dtype(-1.0);
        simi_s_data[j*num+i] = Dtype(-1.0);
        num_conflict++;
      }
      else if(simi_s_data[i*num+j]==Dtype(3.0) && simi_s_data[j*num+i]==Dtype(2.0)){
        simi_s_data[i*num+j] = Dtype(-1.0);
        simi_s_data[j*num+i] = Dtype(-1.0);
        num_conflict++;
      }
      /*
      if((simi_s_data[i*num+j]==Dtype(1.0)||simi_s_data[i*num+j]==Dtype(3.0)) && 
        (simi_s_data[j*num+i]==Dtype(1.0)||simi_s_data[j*num+i]==Dtype(3.0))){
        CHECK_EQ(simi_s_data[i*num+j],simi_s_data[j*num+i]);
        num_similar++;
      }
      else if((simi_s_data[i*num+j]==Dtype(0.0)||simi_s_data[i*num+j]==Dtype(2.0)) && 
        (simi_s_data[j*num+i]==Dtype(0.0)||simi_s_data[j*num+i]==Dtype(2.0))){
        CHECK_EQ(simi_s_data[i*num+j],simi_s_data[j*num+i]);
        num_disimilar++;
      }
      else if(simi_s_data[i*num+j]==Dtype(3.0) || simi_s_data[j*num+i]==Dtype(3.0)){
        num_simi_hyper++;
      }
      else if(simi_s_data[i*num+j]==Dtype(2.0) || simi_s_data[j*num+i]==Dtype(2.0)){
        num_disimi_hyper++;
      }
      else{
        num_ignore++;
      }
      */
    }
  }
  /*
  printf("fea_dim:%d, similar:(%d,%d,%d),disimilar:(%d,%d,%d),ignore:%d,conflict:%d\n",
    fea_dim,num_similar,num_simi_hyper,num_similar+num_simi_hyper,
    num_disimilar,num_disimi_hyper,num_disimilar+num_disimi_hyper,
    num_ignore,num_conflict);  
  */
  // Calculate loss
  // theta[i,j] = 1/2*(bi.*bj)
  // l = sum(sij*theta[i,j]-log(1+exp(theta[i,j])))
  //printf("theta_data[(1-20)*num]:\n");
  Dtype lambda = Dtype(0.1);
  Dtype simi_v = Dtype(1.0);
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < num; j++) {
      theta_data[i*num + j] = Dtype(0.5)*caffe_cpu_dot(dim, bottom_data+i*dim, bottom_data+j*dim);
      //CHECK_EQ(simi_s.cpu_data()[i*num+j],simi_s.cpu_data()[j*num+i]);
      if (simi_s_data[i*num+j] != Dtype(-1.0)) {
        //loss -= log(sigmoid(theta.cpu_data()[i*num+j]));
        //loss = loss - (simi_s.cpu_data()[i*num+j]*theta.cpu_data()[i*num+j] - log(Dtype(1.0)+exp(theta.cpu_data()[i*num+j])));
        lambda = (simi_s_data[i*num+j]>=Dtype(2.0)) ? Dtype(0.1) : Dtype(1.0);
        simi_v = (simi_s_data[i*num+j]==Dtype(1.0)||simi_s_data[i*num+j]==Dtype(3.0)) ? Dtype(1.0) : Dtype(0.0);
        loss = loss - lambda * (simi_v*theta.cpu_data()[i*num+j] - logExp(theta.cpu_data()[i*num+j]));
        //printf("%f\n ",loss);
      }
      //else{
        //loss -= log((1. - sigmoid(theta.cpu_data()[i*num+j])));
      //}
      //LOG(INFO)<<"theta:"<<theta.cpu_data()[i*num+j]<<", loss: "<<loss<<"\n";
    }
    //theta_data[i*num + i] = Dtype(0.5)*caffe_cpu_dot(dim, bottom_data+i*dim, bottom_data+i*dim);
    //if(simi_s_data[i*num+i] == Dtype(1.0)){
    //  loss = loss - (simi_s.cpu_data()[i*num+i]*theta.cpu_data()[i*num+i] - log(Dtype(1.0)+exp(theta.cpu_data()[i*num+i])));
    //}
    //if(i%10==0) printf(" %f",loss);
  }
  //printf("\n");
  //printf("total loss: %f,",loss);
  loss = loss / static_cast<Dtype>(num);
  top[0]->mutable_cpu_data()[0] = loss;
  printf(" ----- ave loss: %f\n",top[0]->cpu_data()[0]);
}


template <typename Dtype>
void SemiPairLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* simi_s_data = simi_s.cpu_data();
    const Dtype* theta_data = theta.cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    int count = bottom[0]->count();
    int num = bottom[0]->num();
    int dim = count/num;
    //Dtype one = Dtype(1.0);
    // Back propagation for each image
    // b[i,j] = 0.5*(sum(1/(1+exp(-theta[i,j])-s[i,j])*u(j)+sum(1/(1+exp(-theta[j,i])))
    Dtype lambda,simi_v;
    for (int i = 0; i<num; i++){
      Blob<Dtype> temp(dim,1,1,1);
      memset(temp.mutable_cpu_data(),0,dim*sizeof(Dtype));
      for (int j = 0; j<num; j++) {
        CHECK_EQ(theta_data[i*num+j],theta_data[j*num+i]);
        //CHECK_EQ(simi_s_data[i*num+j],simi_s_data[j*num+i]);
        if(simi_s_data[i*num+j]!=Dtype(-1.0)){
          //Dtype formerFactor = one/(one+exp(-theta_data[i*num+j]))-simi_s_data[i*num+j];
          //Dtype latterFactor = one/(one+exp(-theta_data[j*num+i]))-simi_s_data[j*num+i];
          lambda = (simi_s_data[i*num+j]>=Dtype(2.0)) ? Dtype(0.1) : Dtype(1.0);
          simi_v = (simi_s_data[i*num+j]==Dtype(1.0)||simi_s_data[i*num+j]==Dtype(3.0)) ? Dtype(1.0) : Dtype(0.0);
          Dtype formerFactor = sigmoid(theta_data[i*num+j])-simi_v;
          caffe_cpu_axpby(dim,lambda*formerFactor,bottom_data+j*dim,Dtype(1.0),temp.mutable_cpu_data());
        }
        if(simi_s_data[j*num+i]!=Dtype(-1.0)){
          lambda = (simi_s_data[j*num+i]>=Dtype(2.0)) ? Dtype(0.1) : Dtype(1.0);
          simi_v = (simi_s_data[j*num+i]==Dtype(1.0)||simi_s_data[i*num+j]==Dtype(3.0)) ? Dtype(1.0) : Dtype(0.0);
          Dtype latterFactor = sigmoid(theta_data[j*num+i])-simi_v;
          caffe_cpu_axpby(dim,lambda*latterFactor,bottom_data+j*dim,Dtype(1.0),temp.mutable_cpu_data());
        }
      }
      caffe_cpu_axpby(dim,Dtype(0.5),temp.cpu_data(),Dtype(0.0),bottom_diff+i*dim);
    }
    caffe_scal(count, top[0]->cpu_diff()[0]/Dtype(num), bottom_diff);
    //LOG(ERROR)<<"diff:"<<bottom_diff[0]<<" "<<bottom_diff[1]<<"\n";
    printf("diff:%f %f\n",bottom_diff[0],bottom_diff[1]);
}

#ifdef CPU_ONLY
STUB_GPU(SemiPairLossLayer);
#endif

INSTANTIATE_CLASS(SemiPairLossLayer);
REGISTER_LAYER_CLASS(SemiPairLoss);

}  // namespace caffe
