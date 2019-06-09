#!/usr/bin/env bash

rm anomalies/*
# -hide_banner -loglevel panic \
/usr/bin/ffmpeg \
    -i $1 \
    -vcodec rawvideo \
    -pix_fmt rgb24 \
    -f image2pipe - 2>stream/input_ffmpeg.stderr |\
    python path_stats.py 2> stream/streetstream.stderr