# Introduction
This is the source code of our TMM 2018 paper "Query-adaptive Image Retrieval by Deep Weighted Hashing", Please cite the following paper if you use our code.

Jian Zhang and Yuxin Peng, "Query-adaptive Image Retrieval by Deep Weighted Hashing", IEEE Transactions on Multimedia (TMM), 2018.
[[PDF]](http://59.108.48.34/mipl/tiki-download_file.php?fileId=351)

# Dependency
Our code is based on early version of [Caffe](https://github.com/BVLC/caffe), all the dependencies are the same as Caffe.

The proposed SSDH also uses the Pre-trained VGG-19 model, which can be downloaded at [Caffe model zoo](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md), download this model and put it in example/qadwh/Pre_trained folder.

# Data Preparation
Here we use MIRFlickr dataset for an example, under "data/flickr25k" folder, there are two list, you should resize MIRFlickr dataset according to those two list, so that Caffe can read the image data.
The flickr25k_triplet_rs15w_noconflict_sf_multi_ label_h5.list is list of h5 file path. The h5 file contains multi label for the images, which is denoted as a vector, for example, (0,0,0,1,1,0,0,0,0,0,0,0,1,0 ...). For more details, see the code of the multi_label layer.

# Usage

1. Edit Makefile.config to suit your system
2. Compile code: make all -j8
3. Training the model: example/qadwh/train_wtri.sh. You may change train_wtri.sh to adjust the parameters such as GPU id and model saving location.
4. Generate hash codes for testing set: example/qadwh/extratfea_txt_flickr25k_12bit.sh. You can adjust script to change hash code saving location GPU id etc.
5. Extract the learned weights for hash functions: example/qadwh/extract_param.sh. You should change parameters in the script such as model locations etc.

For more information, please refer to our [TMM paper](http://59.108.48.34/mipl/tiki-download_file.php?fileId=351).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.
