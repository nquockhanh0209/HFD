#!/usr/bin/bash
# for (( i=2; i <= 4; i++ ))
# do 
#     mkdir Videos/ADL/cam$i
#     mkdir Videos/Fall/cam$i
# done

for (( i=1; i <= 4; i++ ))
do
    cp GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/Subject\ $i/ADL/* Videos/ADL/cam$i/
    cp GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/Subject\ $i/Fall/* Videos/Fall/cam$i/

done
