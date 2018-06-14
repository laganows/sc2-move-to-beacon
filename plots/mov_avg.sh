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
	OUTPUT_FILE=$(echo $OUT_DIR/"${file%.*}".dat)
	yes "0.0" | head -n $((RANGE-1)) > $OUTPUT_FILE
	cat $file | mavg >> $OUTPUT_FILE
	# echo "${file%.*}".dat
done
popd

