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
     void SemiRankLossLayer<Dtype>::LayerSetUp(
       const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::LayerSetUp(bottom, top);
        int dim = bottom[0]->count()/bottom[0]->num();
        int count = bottom[0]->count();
        int num = bottom[0]->num();
        int fea_index = this->layer_param_.semirank_loss_param().feature_index();
        CHECK(fea_index==0||fea_index==2);
        LOG(INFO)<<"margin_rank: "<<this->layer_param_.semirank_loss_param().margin_rank()
          <<" margin_pair_neg: "<<this->layer_param_.semirank_loss_param().margin_pair_neg()
          <<" --- feature_index: "<<this->layer_param_.semirank_loss_param().feature_index()<<"\n";
        LOG(INFO)<<"lambda: "<<this->layer_param_.semirank_loss_param().lambda()
          <<" --- w_pair: "<<this->layer_param_.semirank_loss_param().weight_pair()
          <<" --- w_rank: "<<this->layer_param_.semirank_loss_param().weight_rank()<<"\n";
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
        simi_s.Reshape(num,num,1,1);
        dist_sq_.Reshape(num,num,1,1);
        fea_diff_.Reshape(bottom[fea_index]->channels(),1,1,1);
        data_diff_.Reshape(bottom[0]->channels(),1,1,1);
	iter_count = 0;
     }

   template <typename Dtype>
     void SemiRankLossLayer<Dtype>::Forward_cpu(
       const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top) {
        printf("Forward_cpu... ");
	iter_count++;
        //int count = bottom[0]->count();
        int num = bottom[0]->num();
        int dim = bottom[0]->channels();
        int fea_index = this->layer_param_.semirank_loss_param().feature_index();
        //fea_index = 0;
        int fea_dim = bottom[fea_index]->channels();
        printf("fea_dim:%d -- ",fea_dim);
        
        //reset
        caffe_set(dist_sq_.count(),Dtype(0.0),dist_sq_.mutable_cpu_data());
        caffe_set(simi_s.count(),Dtype(-1.0),simi_s.mutable_cpu_data());
        
        vec_pair_anchor.clear();
        vec_pair_pos.clear();
        vec_pair_neg.clear();
        vec_rank_anchor.clear();
        vec_rank_pos.clear();
        vec_rank_neg.clear();

        const Dtype* bottom_data = bottom[0]->cpu_data(); // bottom_data is n*dim matrix
        const Dtype* bottom_label = bottom[1]->cpu_data();
        const Dtype* bottom_fea = bottom[fea_index]->cpu_data();
        const Dtype* bottom_gt = bottom[3]->cpu_data();
	
	Dtype* dist_sq_data = dist_sq_.mutable_cpu_data();
	Dtype knn_rate = Dtype(0.0);
	Dtype dis_1(0.0),dis_2(0.0),dis_3(0.0),dis_4(0.0),dis_5(0.0);
  	Dtype max2=0.0,min2=20.0,max3=0.0,min3=20.0;

        // pairwide and triplet sampling
        int semipair_count = 0;
	int semipair_count_true = 0;
	int semipair_count_pos_true = 0;
	int semipair_count_neg_true = 0;
        for(int i = 0; i<num; i++){
          vector < pair<Dtype,int> > simi_vec;
          simi_vec.clear();
          vector<int> neg_idx_vec;
          vector<int> pos_idx_vec;
          for(int j=0;j<num;j++){
            //compute data distance
            caffe_sub(dim,bottom_data+i*dim,bottom_data+j*dim,data_diff_.mutable_cpu_data());
            dist_sq_data[i*num+j] = caffe_cpu_dot(dim,data_diff_.cpu_data(),data_diff_.cpu_data());

            //compute L2 similarity based on feature
            //if (bottom_label[i] == Dtype(-1.0)) {
              caffe_sub(fea_dim,bottom_fea+i*fea_dim,bottom_fea+j*fea_dim,fea_diff_.mutable_cpu_data());
              Dtype fea_dist = caffe_cpu_dot(fea_dim,fea_diff_.cpu_data(),fea_diff_.cpu_data());
              pair<Dtype, int> pitem(fea_dist,j);
              simi_vec.push_back(pitem);
            //}
            if(bottom_label[i] > Dtype(-1.0)){
              if(bottom_label[j]==Dtype(-1.0)) continue;
              if (bottom_label[i] == bottom_label[j]){
                //simi_s_data[i*num+j] = Dtype(1.0);
                pos_idx_vec.push_back(j);
              }
              else{
                //simi_s_data[i*num+j] = Dtype(0.0);
                neg_idx_vec.push_back(j);
              }
            }
          }
	
          int pos_idx, neg_idx;
          //if(bottom_label[i] == Dtype(-1.0)){
	    semipair_count++;
            std::sort(simi_vec.begin(),simi_vec.end());
            int length = static_cast<int>(simi_vec.size())/3;
            //select semi-pair examples
            vec_pair_anchor.push_back(i);
	    //min(10,length) --> min(5,length) 2016/04/27
	    int knn_k = this->layer_param_.semirank_loss_param().knn_k();
	    int knn_count_true = 0;
	    int k_length = std::min(knn_k,length);
	    for(int ki=std::min(knn_k,length)-1;ki>=0;ki--){
	      if(bottom_gt[i]==bottom_gt[simi_vec[ki].second]) knn_count_true++;
	    }
	    knn_rate += Dtype(knn_count_true)/std::min(knn_k,length);
	    dis_1+=simi_vec[k_length].first;
    	    dis_2+=simi_vec[length].first; max2 = std::max(max2,simi_vec[length].first);min2 = std::min(min2,simi_vec[length].first);
    	    dis_3+=simi_vec[simi_vec.size()/2].first; max3 = std::max(max3,simi_vec[simi_vec.size()/2].first);min3 = std::min(min3,simi_vec[simi_vec.size()/2].first);
    	    dis_4+=simi_vec[length*2].first;
    	    dis_5+=simi_vec[simi_vec.size()-1].first;

	    pos_idx = simi_vec[caffe_rng_rand()%std::min(knn_k,length)].second;
	    neg_idx = simi_vec[length*2 + caffe_rng_rand()%length].second;
            vec_pair_pos.push_back(pos_idx);
            vec_pair_neg.push_back(neg_idx);
	    CHECK(bottom_gt[i]>=Dtype(0.0)&&bottom_gt[pos_idx]>=0&&bottom_gt[neg_idx]>=0);
            if(bottom_gt[i]==bottom_gt[pos_idx]){
	      semipair_count_pos_true++;
	    }
	    if(bottom_gt[i]!=bottom_gt[neg_idx]){
	      semipair_count_neg_true++;
	    }
	    if(bottom_gt[i]==bottom_gt[pos_idx]&&bottom_gt[i]!=bottom_gt[neg_idx]) semipair_count_true++;
          //}
          //else{
	  if(bottom_label[i]>Dtype(-1.0)){
            int posnum = pos_idx_vec.size();
            int negnum = neg_idx_vec.size();
            if(posnum<=0) continue;
	    if(negnum<=0) continue;
            //select pair examples
	    /* //2014/04/27 comment the pair selection on labeled data
	    vec_pair_anchor.push_back(i);
            pos_idx = pos_idx_vec[caffe_rng_rand()%posnum];
            CHECK_EQ(bottom_label[i],bottom_label[pos_idx]);
            vec_pair_pos.push_back(pos_idx);
            neg_idx = neg_idx_vec[caffe_rng_rand()%negnum];
            CHECK_GT(bottom_label[neg_idx],Dtype(-1.0));
            CHECK(bottom_label[i]!=bottom_label[neg_idx]);
            vec_pair_neg.push_back(neg_idx);
            */
	    
            //select triplet examples
            vec_rank_anchor.push_back(i);
            pos_idx = pos_idx_vec[caffe_rng_rand()%posnum];
            CHECK_EQ(bottom_label[i],bottom_label[pos_idx]);
            vec_rank_pos.push_back(pos_idx);
            neg_idx = neg_idx_vec[caffe_rng_rand()%negnum];
            CHECK_GT(bottom_label[neg_idx],Dtype(-1.0));
            CHECK(bottom_label[i]!=bottom_label[neg_idx]);
            vec_rank_neg.push_back(neg_idx);
          }
        }
	
        CHECK_EQ(vec_pair_anchor.size(),vec_pair_pos.size());
        CHECK_EQ(vec_pair_anchor.size(),vec_pair_neg.size());
        CHECK_EQ(vec_rank_anchor.size(),vec_rank_pos.size());
        CHECK_EQ(vec_rank_anchor.size(),vec_rank_neg.size());
        
        // Calculate constrastive loss
        Dtype m_pair_neg = this->layer_param_.semirank_loss_param().margin_pair_neg();
        Dtype lambda = this->layer_param_.semirank_loss_param().lambda();
	int iter_threshold = this->layer_param_.semirank_loss_param().iter_threshold();
	if(iter_count<=iter_threshold) lambda = Dtype(0.0);
        Dtype w_pair = this->layer_param_.semirank_loss_param().weight_pair();
        Dtype closs(0.0);
        int pos_idx,neg_idx,anc_idx;
        for(int i=vec_pair_anchor.size()-1;i>=0;i--){
          anc_idx = vec_pair_anchor[i];
          pos_idx = vec_pair_pos[i];
          neg_idx = vec_pair_neg[i];
          Dtype tmploss(0.0);
          tmploss += dist_sq_data[anc_idx*num+pos_idx];
          Dtype dist_d = std::max(static_cast<Dtype>(m_pair_neg - sqrt(dist_sq_data[anc_idx*num+neg_idx])),Dtype(0.0));
          tmploss += dist_d*dist_d;
          if(bottom_label[anc_idx]==Dtype(-1.0)){
            closs += lambda * tmploss;
          }
          else{
            //closs += tmploss;
            closs += lambda * tmploss;
          }
        }
        closs = w_pair * closs / static_cast<Dtype>(vec_pair_anchor.size()) / Dtype(2.0);
        
        // Calculate triplet ranking loss
        Dtype rloss(0.0);
        Dtype m_rank = this->layer_param_.semirank_loss_param().margin_rank();
        Dtype w_rank = this->layer_param_.semirank_loss_param().weight_rank();
        for(int i=vec_rank_anchor.size()-1;i>=0;i--){
          anc_idx = vec_rank_anchor[i];
          pos_idx = vec_rank_pos[i];
          neg_idx = vec_rank_neg[i];
          rloss+=std::max(m_rank+dist_sq_data[anc_idx*num+pos_idx]-dist_sq_data[anc_idx*num+neg_idx],Dtype(0.0));
        }
        rloss = w_rank * rloss / static_cast<Dtype>(vec_rank_anchor.size()) / Dtype(2.0); 
	Dtype rate_all = semipair_count_true/Dtype(semipair_count), rate_pos = semipair_count_pos_true/Dtype(semipair_count), rate_neg = semipair_count_neg_true/Dtype(semipair_count);
        //Calculate ranking loss
        printf("----------------------------------------semipair:%d,true (a%d,p%d,n%d),rate (%.3f,%.3f,%.3f),knn_rate:%.3f --- num_pair:%d, num_rank:%d, closs:%f, rloss:%f --",
          semipair_count,semipair_count_true,semipair_count_pos_true,semipair_count_neg_true,
	  semipair_count_true/Dtype(semipair_count),semipair_count_pos_true/Dtype(semipair_count),semipair_count_neg_true/Dtype(semipair_count),knn_rate/semipair_count,
	  static_cast<int>(vec_pair_anchor.size()),static_cast<int>(vec_rank_anchor.size()),closs,rloss);
        top[0]->mutable_cpu_data()[0] = closs + rloss;
        printf("--total loss: %f\n",top[0]->cpu_data()[0]);
	printf("Forword cpu...-----------------dis_1:%f, dis_2:%f (%f,%f), dis_3:%f (%f,%f),dis_4:%f,dis_5:%f\n",dis_1/semipair_count,dis_2/semipair_count,min2,max2,
	    dis_3/semipair_count, max3, min3,dis_4/semipair_count,dis_5/semipair_count);

	if(iter_count%10==0)
	  LOG(INFO)<<"Iteration "<<iter_count<<", semipair:"<<semipair_count<<",true (a"<<semipair_count_true<<",p"<<semipair_count_pos_true<<",n"<<semipair_count_neg_true<<"), rate ("<<rate_all<<","<<rate_pos<<","
		  <<rate_neg<<") knn_rate:"<<knn_rate/semipair_count;
        //CHECK_EQ(1,0);
     }


   template <typename Dtype>
     void SemiRankLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        printf("Backward_cpu... ");
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* bottom_label = bottom[1]->cpu_data();
        const Dtype* dist_sq_data = dist_sq_.cpu_data();

        int count = bottom[0]->count();
        int num = bottom[0]->num();
        int dim = count/num;
        
        CHECK_EQ(vec_pair_anchor.size(),vec_pair_pos.size());
        CHECK_EQ(vec_pair_anchor.size(),vec_pair_neg.size());
        CHECK_EQ(vec_rank_anchor.size(),vec_rank_pos.size());
        CHECK_EQ(vec_rank_anchor.size(),vec_rank_neg.size());

        caffe_set(count,Dtype(0.0),bottom_diff);
        int anc_idx,pos_idx,neg_idx;
        
        Dtype w_pair = this->layer_param_.semirank_loss_param().weight_pair();
        Dtype m_pair_neg = this->layer_param_.semirank_loss_param().margin_pair_neg();
        Dtype lambda = this->layer_param_.semirank_loss_param().lambda();
	int iter_threshold = this->layer_param_.semirank_loss_param().iter_threshold();
	if(iter_count<=iter_threshold) lambda = Dtype(0.0);
        Dtype alpha = w_pair * top[0]->cpu_diff()[0] / static_cast<Dtype>(vec_pair_anchor.size());
        
        //calculate contrastive gradient
	int pair_valid = 0;
        for(int i=vec_pair_anchor.size()-1;i>=0;i--){
          anc_idx = vec_pair_anchor[i];
          pos_idx = vec_pair_pos[i];
          neg_idx = vec_pair_neg[i];
          Dtype gama = alpha;
          //if(bottom_label[anc_idx]==Dtype(-1.0)){
            gama = lambda * alpha;
          //}
          //else{
          //  CHECK_EQ(bottom_label[anc_idx],bottom_label[pos_idx]);
          //  CHECK(bottom_label[anc_idx]!=bottom_label[neg_idx]);
          //}
          //for the similar pair
          caffe_sub(dim,bottom_data+anc_idx*dim,bottom_data+pos_idx*dim,data_diff_.mutable_cpu_data());
          caffe_cpu_axpby(dim,gama,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+anc_idx*dim);
          caffe_cpu_axpby(dim,-gama,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+pos_idx*dim);
          //for the dissimilar pair
          caffe_sub(dim,bottom_data+anc_idx*dim,bottom_data+neg_idx*dim,data_diff_.mutable_cpu_data());
          Dtype dist = sqrt(dist_sq_data[anc_idx*num+neg_idx]);
          Dtype mdist = m_pair_neg - dist;
          Dtype beta = -gama * mdist / (dist + Dtype(1e-4));
          if(mdist>Dtype(0.0)){
	    pair_valid++;
            caffe_cpu_axpby(dim,beta,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+anc_idx*dim);
            caffe_cpu_axpby(dim,-beta,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+neg_idx*dim);
          }
        }

        //calculate triplet ranking gradient
        Dtype m_rank = this->layer_param_.semirank_loss_param().margin_rank();
        Dtype w_rank = this->layer_param_.semirank_loss_param().weight_rank();
        if(static_cast<int>(vec_rank_anchor.size())>0)
           alpha = w_rank * top[0]->cpu_diff()[0] / static_cast<Dtype>(vec_rank_anchor.size());
        else
           alpha = Dtype(0.0);
        int rank_valid = 0;
        for(int i = vec_rank_anchor.size()-1;i>=0;i--){
          anc_idx = vec_rank_anchor[i];
          pos_idx = vec_rank_pos[i];
          neg_idx = vec_rank_neg[i];
          CHECK(bottom_label[anc_idx]>Dtype(-1.0));
          CHECK_EQ(bottom_label[anc_idx],bottom_label[pos_idx]);
          CHECK(bottom_label[neg_idx]>Dtype(-1.0));
          CHECK(bottom_label[pos_idx]!=bottom_label[neg_idx]);
          if(m_rank + dist_sq_data[anc_idx*num+pos_idx]-dist_sq_data[anc_idx*num+neg_idx]>Dtype(0.0)){
            rank_valid++;
            //for the anchor example --1
            caffe_sub(dim,bottom_data+anc_idx*dim,bottom_data+pos_idx*dim,data_diff_.mutable_cpu_data());
            caffe_cpu_axpby(dim, alpha,data_diff_.cpu_data(),
              Dtype(1.0),bottom_diff+anc_idx*dim);
            //for the positive example
            caffe_cpu_axpby(dim,-alpha,data_diff_.cpu_data(),
              Dtype(1.0),bottom_diff+pos_idx*dim);
            //for the anchor example --2
            caffe_sub(dim,bottom_data+anc_idx*dim,bottom_data+neg_idx*dim,data_diff_.mutable_cpu_data());
            caffe_cpu_axpby(dim,-alpha,data_diff_.cpu_data(),
              Dtype(1.0),bottom_diff+anc_idx*dim);
            //for the negative example
            caffe_cpu_axpby(dim, alpha,data_diff_.cpu_data(),
              Dtype(1.0),bottom_diff+neg_idx*dim);
          }
        }
        printf("-----------------------------------------------------------------lambda: %.3f ---------- pair_valid:%d,rank_valid:%d\n",lambda, pair_valid,rank_valid);
	printf("Backward_cpu... diff:%f %f %f %f\n",bottom_diff[0],bottom_diff[1],bottom_diff[dim-1],bottom_diff[count-1]);
     }

#ifdef CPU_ONLY
   STUB_GPU(SemiRankLossLayer);
#endif

   INSTANTIATE_CLASS(SemiRankLossLayer);
   REGISTER_LAYER_CLASS(SemiRankLoss);

}  // namespace caffe
