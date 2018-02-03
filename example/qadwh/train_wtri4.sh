#!/usr/bin/env sh

TOOLS=./build/tools

#train flickr25k
$TOOLS/caffe train --solver=examples/qadwh/flickr25k_wtri_solver4.prototxt --weights=examples/qadwh/Pre_trained/VGG_ILSVRC_19_layers.caffemodel --log_dir=examples/qadwh/48bit_log/flickr25k --gpu=3

