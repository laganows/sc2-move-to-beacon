#!/usr/bin/env bash

INPUT_DIR=$1
if [ -z $INPUT_DIR ];
then
	    INPUT_DIR="input"
fi

./mov_avg.sh $INPUT_DIR
./std_dev.sh $INPUT_DIR
./plot.sh $INPUT_DIR
