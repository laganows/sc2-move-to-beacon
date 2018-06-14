#!/usr/bin/env bash

DIR_NAME=$1
OUT_DIR=plot
PROJ_HOME=$PWD

plot() {
	FILE=$1
	FORMAT=$2
	PLOT_FILE=$(echo $OUT_DIR/"${FILE%.*}".$FORMAT)
	# OUT_FILE=$(echo ../"${FILE%.*}".txt)
	# TITLE=$(cat $OUT_FILE | head -n 1)
	TITLE=$FILE
	STD_DEV=$(cat stddev/$FILE)
	echo "set term $FORMAT size 960, 600"
	echo "set title '$TITLE'"
	echo "set grid"
	echo "set xlabel 'Attempt'"
	echo "set ylabel 'Reward'"
	echo "set label 1 '{/Symbol s}=$STD_DEV'"
	echo "set label 1 at graph 0.02, 0.85 tc lt 3"
	echo "set output '$PLOT_FILE'"
	echo "plot '$FILE' title 'Reward' w l, 'ma/$FILE' title 'Moving average' w l lw 0.5 lc rgb 'orange'"
}

pushd $DIR_NAME
[ -d $OUT_DIR ] || mkdir $OUT_DIR
for file in *.dat; do
	[ -f "$file" ] || break
	plot $file png | gnuplot -persist
	plot $file svg | gnuplot -persist
done
popd

