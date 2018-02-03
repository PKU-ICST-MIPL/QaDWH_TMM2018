#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ClassWiseProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ClassWiseProductLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(48, 12, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(48, 1, 1, 1)),
        blob_top_whcode_(new Blob<Dtype>(48, 12, 1, 1)) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0.0);
    filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 10;  // 0 or 1
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_whcode_);
  }
  virtual ~ClassWiseProductLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_whcode_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_whcode_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ClassWiseProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(ClassWiseProductLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ClassWiseProductParameter* classwise_product_param = layer_param.mutable_classwise_product_param();
  classwise_product_param->set_num_output(12);
  classwise_product_param->set_num_class(10);
  classwise_product_param->mutable_weight_filler()->set_type("constant");
  classwise_product_param->mutable_weight_filler()->set_value(1);
  classwise_product_param->mutable_bias_filler()->set_type("constant");
  classwise_product_param->mutable_bias_filler()->set_value(0);
  ClassWiseProductLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check the gradient for the first 5 bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}
}  // namespace caffe
