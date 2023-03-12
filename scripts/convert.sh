#!/bin/bash

infolder=$1
outfolder=$2
mkdir -p $outfolder


find "$infolder" -type f -name "*.py" | while read file; do
  outfile="${outfolder}/$file"
  dir_path=$(dirname $outfile)
  mkdir -p "$dir_path"
  python replace.py $file $outfile

  echo "${file},  $infolder"
done