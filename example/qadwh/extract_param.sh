BITNUM=$1
NET_PROTO=examples/qadwh/configure/flickr25k/flickr25k_wtri_feature_vgg19_cw_${BITNUM}b.prototxt
LAYER_NAME=weight_hash
NET_MODEL_TYPE=vgg19_cw_rs15w_ft_gaussian
NET_MODEL_DIR=/media/sdb1/junchao/qadwh/wtri/model/flickr25k/${BITNUM}bit/${NET_MODEL_TYPE}
NET_MODEL_PREFFIX=flickr25k_wtri

NET_MODEL_ITER=2000
./build/examples/qadwh/extract_param.bin $NET_PROTO ${NET_MODEL_DIR}/${NET_MODEL_PREFFIX}_iter_${NET_MODEL_ITER}.caffemodel ${LAYER_NAME} ${NET_MODEL_DIR}/${BITNUM}bit_${NET_MODEL_TYPE}_${NET_MODEL_ITER}_${LAYER_NAME}

NET_MODEL_ITER=7000
./build/examples/qadwh/extract_param.bin $NET_PROTO ${NET_MODEL_DIR}/${NET_MODEL_PREFFIX}_iter_${NET_MODEL_ITER}.caffemodel ${LAYER_NAME} ${NET_MODEL_DIR}/${BITNUM}bit_${NET_MODEL_TYPE}_${NET_MODEL_ITER}_${LAYER_NAME}

NET_MODEL_ITER=19000
./build/examples/qadwh/extract_param.bin $NET_PROTO ${NET_MODEL_DIR}/${NET_MODEL_PREFFIX}_iter_${NET_MODEL_ITER}.caffemodel ${LAYER_NAME} ${NET_MODEL_DIR}/${BITNUM}bit_${NET_MODEL_TYPE}_${NET_MODEL_ITER}_${LAYER_NAME}

