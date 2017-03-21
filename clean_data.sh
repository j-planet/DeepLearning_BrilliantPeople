#!/usr/bin/env bash

sed -i backup 's/^people//g' ./data/names.txt
sed -i backup 's/^king//g' ./data/names.txt
sed -i backup 's/^historical//g' ./data/names.txt
sed -i backup 's/historical (/ (/g' ./data/names.txt
sed -i backup 's/(historical/(/g' ./data/names.txt
