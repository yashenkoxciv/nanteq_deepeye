#!/usr/bin/env bash

rm stream/*
# -hide_banner -loglevel panic \
/usr/bin/ffmpeg \
    -i $1 \
    -vcodec rawvideo \
    -pix_fmt rgb24 \
    -f image2pipe - 2>stream/input_ffmpeg.stderr |\
    python streetstream.py 2> stream/streetstream.stderr |\
    /usr/bin/ffmpeg \
        -f rawvideo \
        -pixel_format rgb24 \
        -video_size 1920x1080 \
        -framerate 24 \
        -i - \
        -f mpeg \
        video.mp4


