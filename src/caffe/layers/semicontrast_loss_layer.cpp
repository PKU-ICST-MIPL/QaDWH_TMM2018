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
void SemiContrastLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  int dim = bottom[0]->count()/bottom[0]->num();
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int fea_index = this->layer_param_.semicontrast_loss_param().feature_index();
  CHECK(fea_index==0||fea_index==2);
  LOG(INFO)<<"margin: "<<this->layer_param_.semicontrast_loss_param().margin()
    <<"--- margin_sup: "<<this->layer_param_.semicontrast_loss_param().margin_sup()
    <<" --- lambda: "<<this->layer_param_.semicontrast_loss_param().lambda()
    <<" --- feature_index: "<<this->layer_param_.semicontrast_loss_param().feature_index()<<"\n";
  LOG(INFO)<<"count: "<<count<<" --- num: "<<num<<" --- dim: "<<dim<<"--- feature dim: "<<bottom[fea_index]->channels()<<"\n";
  CHECK_EQ(bottom[0]->channels(), dim);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->num(),bottom[0]->num());
  CHECK_EQ(bottom[2]->height(),1);
  CHECK_EQ(bottom[2]->width(),1);
  iter_count = 0;
  simi_s.Reshape(num,num,1,1);
  dist_sq_.Reshape(num,num,1,1);
  fea_diff_.Reshape(bottom[fea_index]->channels(),1,1,1);
  data_diff_.Reshape(bottom[0]->channels(),1,1,1);
}

template <typename Dtype>
void SemiContrastLossLayer<Dtype>::Forward_cpu(
const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  iter_count++;
  printf("Forward_cpu... ");
  //int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = bottom[0]->channels();
  int fea_index = this->layer_param_.semicontrast_loss_param().feature_index();
  int fea_dim = bottom[fea_index]->channels();
  int knn_k = this->layer_param_.semicontrast_loss_param().knn_k();
 
  //reset
  caffe_set(dist_sq_.count(),Dtype(0.0),dist_sq_.mutable_cpu_data());
  vec_sup_anchor.clear();
  vec_sup_pos.clear();
  vec_sup_neg.clear();
  vec_semi_anchor.clear();
  vec_semi_pos.clear();
  vec_semi_neg.clear();

  const Dtype* bottom_data = bottom[0]->cpu_data(); // bottom_data is n*dim matrix
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* bottom_fea = bottom[fea_index]->cpu_data();
  const Dtype* bottom_gt = bottom[3]->cpu_data();

  Dtype dis_1(0.0),dis_2(0.0),dis_3(0.0),dis_4(0.0),dis_5(0.0);
  Dtype max2=0.0,min2=20.0,max3=0.0,min3=20.0;
  Dtype* dist_sq_data = dist_sq_.mutable_cpu_data();
  Dtype knn_rate = Dtype(0.0);
  //supervise sampling and supervise sampling
  int semipair_count = 0;
  int semipair_count_true = 0,semipair_count_pos_true = 0,semipair_count_neg_true = 0;
  int anc_idx,pos_idx,neg_idx;
  for(int i = 0; i<num; i++){
    vector < pair<Dtype,int> > simi_vec;
    simi_vec.clear();
    vector<int> neg_idx_vec;
    vector<int> pos_idx_vec;
    for(int j=0;j<num;j++){
      //compute L2 similarity based on feature
      caffe_sub(fea_dim,bottom_fea+i*fea_dim,bottom_fea+j*fea_dim,fea_diff_.mutable_cpu_data());
      Dtype fea_dist = caffe_cpu_dot(fea_dim,fea_diff_.cpu_data(),fea_diff_.cpu_data());
      pair<Dtype, int> pitem(fea_dist,j);
      simi_vec.push_back(pitem);
      //compute data distance
      caffe_sub(dim,bottom_data+i*dim,bottom_data+j*dim,data_diff_.mutable_cpu_data());
      dist_sq_data[i*num+j] = caffe_cpu_dot(dim,data_diff_.cpu_data(),data_diff_.cpu_data());
      
      if(bottom_label[i] > Dtype(-1.0)){
        if(bottom_label[j]==Dtype(-1.0)) continue;
        if (bottom_label[i] == bottom_label[j]){
          pos_idx_vec.push_back(j);
        }
        else{
          neg_idx_vec.push_back(j);
        }
      }
    }
    //select semi pairs
    std::sort(simi_vec.begin(),simi_vec.end());
    vec_semi_anchor.push_back(i);
    int length = static_cast<int>(simi_vec.size())/3;
    int knn_count_true = 0;
    int k_length = std::min(knn_k,length);
    
    for(int ki=k_length-1;ki>=0;ki--){
      if(bottom_gt[i]==bottom_gt[simi_vec[ki].second]) knn_count_true++;
    }
    knn_rate += Dtype(knn_count_true)/k_length;
    dis_1+=simi_vec[k_length].first;
    dis_2+=simi_vec[length].first; max2 = std::max(max2,simi_vec[length].first);min2 = std::min(min2,simi_vec[length].first);
    dis_3+=simi_vec[simi_vec.size()/2].first; max3 = std::max(max3,simi_vec[simi_vec.size()/2].first);min3 = std::min(min3,simi_vec[simi_vec.size()/2].first);
    dis_4+=simi_vec[length*2].first;
    dis_5+=simi_vec[simi_vec.size()-1].first;
    pos_idx = simi_vec[caffe_rng_rand()%k_length].second;
    //neg_idx = simi_vec[2*k_length + caffe_rng_rand()%(length-2*k_length)].second;
    int idx_start = this->layer_param_.semicontrast_loss_param().neg_idx_start();
    int idx_range = this->layer_param_.semicontrast_loss_param().neg_idx_range();
    neg_idx = simi_vec[idx_start + caffe_rng_rand()%idx_range].second;
    vec_semi_pos.push_back(pos_idx);
    vec_semi_neg.push_back(neg_idx);
    //statistic
    int true_pair = 1;
    if(bottom_gt[i]==bottom_gt[pos_idx]) semipair_count_pos_true++;
    else true_pair*=0;
    if(bottom_gt[i]!=bottom_gt[neg_idx]) semipair_count_neg_true++;
    else true_pair*=0;
    semipair_count_true+=true_pair;

    //select sup pairs
    if (bottom_label[i]==Dtype(-1.0)) continue;
    int posnum = pos_idx_vec.size();
    int negnum = neg_idx_vec.size();
    if(posnum<=0 || negnum<=0) continue;
    CHECK_GT(bottom_label[i],Dtype(-1.0));
    
    vec_sup_anchor.push_back(i);
    pos_idx = pos_idx_vec[caffe_rng_rand()%posnum];
    CHECK_EQ(bottom_label[i],bottom_label[pos_idx]);
    vec_sup_pos.push_back(pos_idx);
    neg_idx = neg_idx_vec[caffe_rng_rand()%negnum];
    CHECK(bottom_label[i]!=bottom_label[neg_idx]);
    vec_sup_neg.push_back(neg_idx);
  }
  semipair_count = static_cast<int>(vec_semi_anchor.size());
  // Calculate constrastive loss
  Dtype margin = this->layer_param_.semicontrast_loss_param().margin_sup();
  Dtype lambda = this->layer_param_.semicontrast_loss_param().lambda();
  
  int iter_threshold = this->layer_param_.semicontrast_loss_param().iter_threshold();
  if (iter_count<=iter_threshold) lambda = Dtype(0.0);
  //sup loss
  Dtype sup_loss(0.0);
  for(int i = vec_sup_anchor.size()-1;i>=0;i--){
    anc_idx = vec_sup_anchor[i];
    pos_idx = vec_sup_pos[i];
    neg_idx = vec_sup_neg[i];
    sup_loss = dist_sq_data[anc_idx*num+pos_idx];
    Dtype dist_d = std::max(static_cast<Dtype>(margin-sqrt(dist_sq_data[anc_idx*num+neg_idx])),Dtype(0.0));
    sup_loss+=dist_d*dist_d;
  } 
  sup_loss = sup_loss / static_cast<int>(vec_sup_anchor.size()) / Dtype(2.0);
  //semi loss
  Dtype semi_loss(0.0);
  margin = this->layer_param_.semicontrast_loss_param().margin();
  //if (lambda>Dtype(0.0)){
    for(int i = vec_semi_anchor.size()-1;i>=0;i--){
      anc_idx = vec_semi_anchor[i];
      pos_idx = vec_semi_pos[i];
      neg_idx = vec_semi_neg[i];
      semi_loss = dist_sq_data[anc_idx*num+pos_idx];
      Dtype dist_d = std::max(static_cast<Dtype>(margin-sqrt(dist_sq_data[anc_idx*num+neg_idx])),Dtype(0.0));
      semi_loss+=dist_d*dist_d;
    }    
  //}  
  semi_loss = lambda * semi_loss / static_cast<int>(vec_semi_anchor.size()) / Dtype(2.0);
  Dtype semi_rate_a =  semipair_count_true/Dtype(semipair_count);
  Dtype semi_rate_p = semipair_count_pos_true/Dtype(semipair_count);
  Dtype semi_rate_n = semipair_count_neg_true/Dtype(semipair_count);
  printf("-----lambda:%.2f, semi_true(a%d,p%d,n%d) rate(%.3f,%.3f.%.3f) knn_rate:%.3f------ num_sup:%d sup_loss:%f, num_semi:%d semi_loss:%f --",
    lambda,semipair_count_true,semipair_count_pos_true,semipair_count_neg_true,
    semi_rate_a,semi_rate_p,semi_rate_n,knn_rate/semipair_count,
    static_cast<int>(vec_sup_anchor.size()),sup_loss,
    static_cast<int>(vec_semi_anchor.size()),semi_loss);

  top[0]->mutable_cpu_data()[0] = sup_loss + semi_loss;
  printf("--ave loss: %f\n",top[0]->cpu_data()[0]);
  printf("Forword cpu...-----------------dis_1:%f, dis_2:%f (%f,%f), dis_3:%f (%f,%f),dis_4:%f,dis_5:%f\n",dis_1/semipair_count,dis_2/semipair_count,min2,max2,
    dis_3/semipair_count, max3, min3,dis_4/semipair_count,dis_5/semipair_count);
  if(iter_count%10==0){
    LOG(INFO)<<"Iteration "<<iter_count<<" lambda:"<<lambda<<" supnum:"<<vec_sup_anchor.size()
      <<" seminum:"<<vec_semi_anchor.size()
      <<" semi_true(a"<<semipair_count_true<<",p"<<semipair_count_pos_true<<",n"<<semipair_count_neg_true
      <<") rate("<<semi_rate_a<<","<<semi_rate_p<<","<<semi_rate_n<<") knn_rate:"<<knn_rate/semipair_count
      <<" sup_loss:"<<sup_loss<<" semi_loss:"<<semi_loss;
  }
  //CHECK_EQ(1,0);
 }


template <typename Dtype>
void SemiContrastLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  printf("Backward_cpu... ");
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* dist_sq_data = dist_sq_.cpu_data();

  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count/num;

  Dtype margin = this->layer_param_.semicontrast_loss_param().margin_sup();
  Dtype lambda = this->layer_param_.semicontrast_loss_param().lambda();
  int iter_threshold = this->layer_param_.semicontrast_loss_param().iter_threshold();
  if(iter_count<=iter_threshold) lambda = Dtype(0.0);
  Dtype alpha = top[0]->cpu_diff()[0] / static_cast<int>(vec_sup_anchor.size());
  caffe_set(count,Dtype(0.0),bottom_diff);
  int anc_idx,pos_idx,neg_idx;
  //calculate contrastive gradient--sup
  int sup_valid = 0;
  for (int i=vec_sup_anchor.size()-1;i>=0;i--){
    anc_idx = vec_sup_anchor[i];
    pos_idx = vec_sup_pos[i];
    neg_idx = vec_sup_neg[i];
    //for pos pair
    caffe_sub(dim,bottom_data+anc_idx*dim,bottom_data+pos_idx*dim,data_diff_.mutable_cpu_data());
    caffe_cpu_axpby(dim,alpha,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+anc_idx*dim);
    caffe_cpu_axpby(dim,-alpha,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+pos_idx*dim);
    //for neg pair
    caffe_sub(dim,bottom_data+anc_idx*dim,bottom_data+neg_idx*dim,data_diff_.mutable_cpu_data());
    Dtype dist = sqrt(dist_sq_data[anc_idx*num+neg_idx]);
    Dtype mdist = margin - dist;
    Dtype beta = -alpha * mdist / (dist + Dtype(1e-4));
    if(mdist>Dtype(0.0)){
      sup_valid++;
      caffe_cpu_axpby(dim,beta,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+anc_idx*dim);
      caffe_cpu_axpby(dim,-beta,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+neg_idx*dim);
    }
  }
  int semi_valid = 0;
  margin = this->layer_param_.semicontrast_loss_param().margin();
  alpha = lambda*top[0]->cpu_diff()[0] / static_cast<int>(vec_semi_anchor.size());
  if(lambda>Dtype(0.0)){
    for (int i=vec_semi_anchor.size()-1;i>=0;i--){
      anc_idx = vec_semi_anchor[i];
      pos_idx = vec_semi_pos[i];
      neg_idx = vec_semi_neg[i];
      //for pos pair
      caffe_sub(dim,bottom_data+anc_idx*dim,bottom_data+pos_idx*dim,data_diff_.mutable_cpu_data());
      caffe_cpu_axpby(dim,alpha,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+anc_idx*dim);
      caffe_cpu_axpby(dim,-alpha,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+pos_idx*dim);
      //for neg pair
      caffe_sub(dim,bottom_data+anc_idx*dim,bottom_data+neg_idx*dim,data_diff_.mutable_cpu_data());
      Dtype dist = sqrt(dist_sq_data[anc_idx*num+neg_idx]);
      Dtype mdist = margin - dist;
      Dtype beta = -alpha * mdist / (dist + Dtype(1e-4));
      if(mdist>Dtype(0.0)){
        semi_valid++;
        caffe_cpu_axpby(dim,beta,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+anc_idx*dim);
        caffe_cpu_axpby(dim,-beta,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+neg_idx*dim);
      }
    }
  }
  else{
    for (int i=vec_semi_anchor.size()-1;i>=0;i--){
      anc_idx = vec_semi_anchor[i];
      pos_idx = vec_semi_pos[i];
      neg_idx = vec_semi_neg[i];
      Dtype dist = sqrt(dist_sq_data[anc_idx*num+neg_idx]);
      Dtype mdist = margin - dist;
      if(mdist>Dtype(0.0)){
        semi_valid++;
      }
    }
  }

  printf("------sup_valid:%d,semi_valid:%d, diff:%f %f %f %f\n",sup_valid,semi_valid,bottom_diff[0],bottom_diff[1],bottom_diff[dim-1],bottom_diff[count-1]);
  if(iter_count%10==0) LOG(INFO)<<"Iteration "<<iter_count<<", sup_valid:"<<sup_valid<<" semi_valid:"<<semi_valid;
}

#ifdef CPU_ONLY
   STUB_GPU(SemiContrastLossLayer);
#endif

INSTANTIATE_CLASS(SemiContrastLossLayer);
REGISTER_LAYER_CLASS(SemiContrastLoss);

}  // namespace caffe
