#!/usr/bin/env bash

DIR_NAME=$1
OUT_DIR=ma
PROJ_HOME=$PWD
RANGE=10

mavg() {
	scala $PROJ_HOME/mov_avg.sc $RANGE
}

pushd $DIR_NAME
[ -d $OUT_DIR ] || mkdir $OUT_DIR
for file in *.dat; do
	[ -f "$file" ] || break
	cat $file | mavg > $(echo $OUT_DIR/"${file%.*}".dat)
	# echo "${file%.*}".dat
done
popd

