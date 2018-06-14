#!/usr/bin/env bash

INPUT_DIR=input

./mov_avg.sh $INPUT_DIR
./std_dev.sh $INPUT_DIR
./plot.sh $INPUT_DIR
