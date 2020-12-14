#!/bin/bash

for name in *.wav;
do
    ffmpeg -i "$name" -c:a copy "tmp/$name"
done