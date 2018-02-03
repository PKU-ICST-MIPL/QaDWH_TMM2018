#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void ClassWiseProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), 2)<< "CP Layer takes two blobs as input: the first one for data and the second one for label.";
	CHECK_EQ(top.size(), 1) << "CP Layer takes a single blob as output.";
	CHECK_EQ(bottom[0]->num(), bottom[1]->num());
	const int num_output = this->layer_param_.classwise_product_param().num_output(); //does we need this ?
	biasterm_ = this->layer_param_.classwise_product_param().bias_term();
	C_ = this->layer_param_.classwise_product_param().num_class();
	// Figure out the dimensions
	M_ = bottom[0]->num();  // number of simples
	K_ = bottom[0]->count() / bottom[0]->num(); // the dimensional of each feature
	N_ = num_output; // in this layer, K_ equals to N_
	CHECK_EQ(K_, N_);
	top[0]->Reshape(bottom[0]->num(), num_output, 1, 1);

	// Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {
		if (biasterm_) {
			this->blobs_.resize(2);
		} else {
			this->blobs_.resize(1);
		}
		// Intialize the weight
		this->blobs_[0].reset(new Blob<Dtype>(C_, K_, 1, 1));
		// fill the weights
		shared_ptr<Filler<Dtype> > weight_filler(
				GetFiller<Dtype>(this->layer_param_.classwise_product_param().weight_filler()));
		weight_filler->Fill(this->blobs_[0].get());
		// If necessary, intiialize and fill the bias term
		if (biasterm_) {
			this->blobs_[1].reset(new Blob<Dtype>(1, N_, 1, 1));
			shared_ptr<Filler<Dtype> > bias_filler(
					GetFiller<Dtype>(this->layer_param_.classwise_product_param().bias_filler()));
			bias_filler->Fill(this->blobs_[1].get());
		}
	}  // parameter initialization

	LOG(INFO)<<"ClassWiseProductLayer setup done: shape ("<<this->blobs_[0]->num()<<", "
	<<this->blobs_[0]->channels()<<", "<<this->blobs_[0]->height()<<", "<<this->blobs_[0]->width()<<"); label dim: "<<bottom[1]->channels();
//	   // Setting up the bias multiplier
//	if (biasterm_) {
//		LOG(INFO) << "111";
//		bias_multiplier_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
//		Dtype* bias_multiplier_data =
//		reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
//		for (int i = 0; i < N_; ++i) {
//			bias_multiplier_data[i] = 1.;
//		}
//	}
};

template <typename Dtype>
void ClassWiseProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}




template<typename Dtype>
void ClassWiseProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	const int num = bottom[0]->num();
	const int dim = bottom[0]->count() / bottom[0]->num();
	CHECK_EQ(dim, this->blobs_[0]->channels());
  int label_dim = static_cast<int>(bottom[1]->channels());
  if(bottom[1]->channels()==1){
	  for ( int i = 0; i < num ; i++ )
	  {
		  const Dtype* data = bottom_data + i* dim;
	 	  int cur_label = static_cast<int>(bottom_label[i]);
		  CHECK_LT(cur_label, C_);
		  const Dtype* cur_weight = weight + cur_label*dim; 		
		  caffe_mul( dim, data, cur_weight, top_data + i*dim );
	  }
  }
  else {
    Blob<Dtype> fusion_weight(num,dim,1,1);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,num,dim,this->blobs_[0]->num(),Dtype(1.),bottom_label,weight,Dtype(0.),fusion_weight.mutable_cpu_data());
    for (int i=0; i<num; i++){
      Blob<Dtype> cur_weight_(1,dim,1,1);
      Dtype* cur_weight = cur_weight_.mutable_cpu_data();
      Dtype sum = caffe_cpu_dot(label_dim, bottom_label+i*label_dim, bottom_label+i*label_dim);
      caffe_cpu_axpby(dim, Dtype(1.)/sum, fusion_weight.cpu_data()+i*dim, Dtype(0.), cur_weight);
		  caffe_mul( dim, bottom_data + i* dim, cur_weight, top_data + i*dim );
    }
  }
}





template<typename Dtype>
void ClassWiseProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	const int num = top[0]->num();
	const int dim = top[0]->count() / top[0]->num();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* weight = this->blobs_[0]->cpu_data();
  int label_dim = static_cast<int>(bottom[1]->channels());

	Blob<Dtype> intermediate_result(1, dim, 1, 1);
	memset(intermediate_result.mutable_cpu_data(), 0, sizeof(Dtype) * dim);
  

  if(label_dim==1){
	// Gradient with respect to weight
	for ( int i = 0; i < num ; i++ )
	{
		const Dtype* diff_data = top_diff + i* dim; 
		const Dtype* data = bottom_data + i* dim;
    const Dtype* bottom_label = bottom[1]->cpu_data();

		caffe_mul( dim, diff_data, data, intermediate_result.mutable_cpu_data() );
		caffe_axpy( dim, Dtype(1), intermediate_result.cpu_data(), this->blobs_[0]->mutable_cpu_diff() + static_cast<int>(bottom_label[i])*dim);
	}

	if (propagate_down[0]) {
		// Gradient with respect to bottom data
		for ( int i = 0; i < num ; i++ )
		{
			const Dtype* diff_data = top_diff + i* dim; 
			Dtype* diff_data_b = bottom_diff + i* dim; 		
			caffe_mul( dim, diff_data, weight + static_cast<int>(bottom_label[i])*dim, diff_data_b );
		}
	}
  }
  else{
    // Gradient with respect to weight
    for ( int i = 0; i < num ; i++ ){
      const Dtype* diff_data = top_diff + i* dim;
      const Dtype* data = bottom_data + i* dim;
      caffe_mul( dim, diff_data, data, intermediate_result.mutable_cpu_data());
      Dtype sum = caffe_cpu_dot(label_dim, bottom_label+i*label_dim, bottom_label+i*label_dim);
      for(int j=0; j<label_dim; j++){
        if(bottom_label[i*label_dim+j]>Dtype(0.)){
          caffe_axpy( dim, Dtype(1.)/sum, intermediate_result.cpu_data(), this->blobs_[0]->mutable_cpu_diff() + j*dim);
        }
      }
    }

    if (propagate_down[0]) {
      Blob<Dtype> fusion_weight(num,dim,1,1);
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,num,dim,this->blobs_[0]->num(),Dtype(1.),bottom_label,weight,Dtype(0.),fusion_weight.mutable_cpu_data());
      for ( int i = 0; i < num ; i++ ){
        const Dtype* diff_data = top_diff + i* dim;
        Dtype* diff_data_b = bottom_diff + i* dim;
        
        Blob<Dtype> cur_weight_(1,dim,1,1);
        Dtype* cur_weight = cur_weight_.mutable_cpu_data();
        Dtype sum = caffe_cpu_dot(label_dim, bottom_label+i*label_dim, bottom_label+i*label_dim);
        caffe_cpu_axpby(dim, Dtype(1.)/sum, fusion_weight.cpu_data()+i*dim, Dtype(0.), cur_weight);
        caffe_mul( dim, diff_data, cur_weight, diff_data_b );
      }
    }
  }
}




#ifdef CPU_ONLY
STUB_GPU(ClassWiseProductLayer);
#endif

INSTANTIATE_CLASS(ClassWiseProductLayer);
REGISTER_LAYER_CLASS(ClassWiseProduct);

}  // namespace caffe
