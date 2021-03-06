#!/usr/bin/env sh

RUN_BIN=./build/tools

PROTO=./examples/qadwh/configure/flickr25k/flickr25k_wtri_feature_vgg19_cw_24b.prototxt
TOTAL_NUM=20000
BATCH_NUM=50
FEA_NUM=`expr $TOTAL_NUM / $BATCH_NUM`
GPU_ID=1
echo "Begin Extract fea"

MODEL_NAME=/media/sdb1/junchao/qadwh/wtri/model/flickr25k/24bit/vgg19_cw_rs15w_ft_gaussian/flickr25k_wtri_iter_2000.caffemodel
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/24bit_features_vgg19_cw_rs15w_ft_gaussian_2K_prob
echo $MODEL_NAME
echo $FEA_DIR
echo $PROTO
echo "Total Feature num: ${FEA_NUM}"
GLOG_logtostderr=0
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} prob ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/24bit_features_vgg19_cw_rs15w_ft_gaussian_2K_pred
echo $FEA_DIR
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} pred ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID

MODEL_NAME=/media/sdb1/junchao/qadwh/wtri/model/flickr25k/24bit/vgg19_cw_rs15w_ft_gaussian/flickr25k_wtri_iter_9000.caffemodel
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/24bit_features_vgg19_cw_rs15w_ft_gaussian_9K_prob
echo $MODEL_NAME
echo $FEA_DIR
echo $PROTO
echo "Total Feature num: ${FEA_NUM}"
GLOG_logtostderr=0
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} prob ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/24bit_features_vgg19_cw_rs15w_ft_gaussian_9K_pred
echo $FEA_DIR
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} pred ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID

MODEL_NAME=/media/sdb1/junchao/qadwh/wtri/model/flickr25k/24bit/vgg19_cw_rs15w_ft_gaussian/flickr25k_wtri_iter_19000.caffemodel
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/24bit_features_vgg19_cw_rs15w_ft_gaussian_19K_prob
echo $MODEL_NAME
echo $FEA_DIR
echo $PROTO
echo "Total Feature num: ${FEA_NUM}"
GLOG_logtostderr=0
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} prob ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/24bit_features_vgg19_cw_rs15w_ft_gaussian_19K_pred
echo $FEA_DIR
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} pred ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID

MODEL_NAME=/media/sdb1/junchao/qadwh/wtri/model/flickr25k/24bit/vgg19_cw_rs15w_ft_gaussian/flickr25k_wtri_iter_22000.caffemodel
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/24bit_features_vgg19_cw_rs15w_ft_gaussian_22K_prob
echo $MODEL_NAME
echo $FEA_DIR
echo $PROTO
echo "Total Feature num: ${FEA_NUM}"
GLOG_logtostderr=0
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} prob ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/24bit_features_vgg19_cw_rs15w_ft_gaussian_22K_pred
echo $FEA_DIR
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} pred ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID

