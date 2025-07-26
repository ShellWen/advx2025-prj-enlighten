#!/usr/bin/env bash
arecord -Dhw:0,0 -c 2 -r 48000 -f S24_LE -t wav -d 1 /tmp/fix.wav
arecord -Dhw:1,0 -c 2 -r 48000 -f S24_LE -t wav -d 1 /tmp/fix.wav
