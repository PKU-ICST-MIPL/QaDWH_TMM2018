#!/usr/bin/env sh

RUN_BIN=./build/tools

PROTO=./examples/qadwh/configure/flickr25k/flickr25k_wtri_feature_vgg19_cw_48b.prototxt
TOTAL_NUM=20000
BATCH_NUM=50
FEA_NUM=`expr $TOTAL_NUM / $BATCH_NUM`
GPU_ID=3
echo "Begin Extract fea"

MODEL_NAME=/media/sdb1/junchao/qadwh/wtri/model/flickr25k/48bit/vgg19_cw_rs15w_ft_gaussian/flickr25k_wtri_iter_2000.caffemodel
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/48bit_features_vgg19_cw_rs15w_ft_gaussian_2K_prob
echo $MODEL_NAME
echo $FEA_DIR
echo $PROTO
echo "Total Feature num: ${FEA_NUM}"
GLOG_logtostderr=0
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} prob ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/48bit_features_vgg19_cw_rs15w_ft_gaussian_2K_pred
echo $FEA_DIR
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} pred ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID

MODEL_NAME=/media/sdb1/junchao/qadwh/wtri/model/flickr25k/48bit/vgg19_cw_rs15w_ft_gaussian/flickr25k_wtri_iter_7000.caffemodel
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/48bit_features_vgg19_cw_rs15w_ft_gaussian_7K_prob
echo $MODEL_NAME
echo $FEA_DIR
echo $PROTO
echo "Total Feature num: ${FEA_NUM}"
GLOG_logtostderr=0
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} prob ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/48bit_features_vgg19_cw_rs15w_ft_gaussian_7K_pred
echo $FEA_DIR
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} pred ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID

MODEL_NAME=/media/sdb1/junchao/qadwh/wtri/model/flickr25k/48bit/vgg19_cw_rs15w_ft_gaussian/flickr25k_wtri_iter_19000.caffemodel
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/48bit_features_vgg19_cw_rs15w_ft_gaussian_19K_prob
echo $MODEL_NAME
echo $FEA_DIR
echo $PROTO
echo "Total Feature num: ${FEA_NUM}"
GLOG_logtostderr=0
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} prob ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID
FEA_DIR=/media/sdb1/junchao/qadwh/wtri/features/flickr25k/48bit_features_vgg19_cw_rs15w_ft_gaussian_19K_pred
echo $FEA_DIR
${RUN_BIN}/extract_features_text.bin ${MODEL_NAME} ${PROTO} pred ${FEA_DIR} ${FEA_NUM} lmdb GPU $GPU_ID

