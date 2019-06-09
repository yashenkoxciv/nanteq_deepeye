#!/usr/bin/env bash

rm stream/*
rm frames/*
# -hide_banner -loglevel panic \
/usr/bin/ffmpeg \
    -i $1 \
    -ss 17\
    -vcodec rawvideo \
    -pix_fmt rgb24 \
    -f image2pipe - 2>stream/input_ffmpeg.stderr |\
    python streetstream.py 2> stream/streetstream.stderr |\
    /usr/bin/ffmpeg \
        -f rawvideo \
        -pixel_format rgb24 \
        -video_size 1920x1080 \
        -i - \
        frames/%06d.jpg


