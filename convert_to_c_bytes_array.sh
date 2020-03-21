#!/bin/sh

xxd -i model.tflite > model.cc
xxd -i model_quantized.tflite > model_quantized.cc