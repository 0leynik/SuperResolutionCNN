#!/usr/bin/env bash

source ~/tensorflow/bin/activate

python evaluate.py -i /Users/dmitryoleynik/PycharmProjects/SuperResolutionCNN/increase_2x
python evaluate.py -p /Users/dmitryoleynik/PycharmProjects/SuperResolutionCNN/predict_2x

deactivate