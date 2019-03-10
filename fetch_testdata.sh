#!/bin/sh

mkdir data/
cd data/
# wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
mv annotations/instances_val2014.json .
rm -r annotations/ annotations_trainval2014.zip
wget https://raw.githubusercontent.com/cocodataset/cocoapi/master/results/instances_val2014_fakebbox100_results.json
