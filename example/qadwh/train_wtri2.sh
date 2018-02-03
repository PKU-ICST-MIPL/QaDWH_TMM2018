#!/usr/bin/env sh

TOOLS=./build/tools

#train flickr25k
$TOOLS/caffe train --solver=examples/qadwh/flickr25k_wtri_solver2.prototxt --weights=examples/qadwh/Pre_trained/VGG_ILSVRC_19_layers.caffemodel --log_dir=examples/qadwh/24bit_log/flickr25k --gpu=1

