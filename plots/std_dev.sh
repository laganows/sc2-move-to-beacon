#!/usr/bin/env bash

DIR_NAME=$1
OUT_DIR=stddev
PROJ_HOME=$PWD

stddev() {
	scala $PROJ_HOME/std_dev.sc
}

pushd $DIR_NAME
[ -d $OUT_DIR ] || mkdir $OUT_DIR
for file in *.dat; do
	[ -f "$file" ] || break
	cat $file | stddev > $(echo $OUT_DIR/"${file%.*}".dat)
	# echo "${file%.*}".dat
done
popd

