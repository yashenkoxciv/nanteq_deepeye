#!/usr/bin/env bash

rm stream/*
# -hide_banner -loglevel panic \
/usr/bin/ffmpeg \
    -i https://s1.worldcam.live:8082/datexnetkurow/tracks-v1/mono.m3u8 \
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
        -f segment \
        -segment_time 10 \
        -segment_format mpegts \
        -segment_list "stream/streetstream.m3u8" \
        -segment_list_type m3u8 \
        "stream/segment%d.ts"


<< --STREAM-TO-PYTHON-TO-MP4--
/usr/bin/ffmpeg \
    -i https://s1.worldcam.live:8082/datexnetkurow/tracks-v1/mono.m3u8 \
    -vcodec rawvideo \
    -pix_fmt rgb24 \
    -f image2pipe - |\
    python streetstream.py |\
    /usr/bin/ffmpeg \
        -f rawvideo \
        -pixel_format rgb24 \
        -video_size 1920x1080 \
        -framerate 24 \
        -i - \
        stream.mp4
--STREAM-TO-PYTHON-TO-MP4--