#!/bin/bash

OUTPUT_FILE=results/
DEVICE=CPU
FP_MODEL=FP16
INPUT_FILE=/data/reference-sample-data/object-detection-python/cars_1900.mp4
NUM_REQS=4

source openvino_dev/bin/activate

# Make sure that the output directory exists.
mkdir -p $OUTPUT_FILE

# Set inference model IR files using specified precision
MODELPATH=models/mobilenet-ssd/${FP_MODEL}/mobilenet-ssd.xml
LABEL_FILE=/data/reference-sample-data/object-detection-python/labels.txt

# Run the object detection code
python3 object_detection.py -m $MODELPATH \
                            -i $INPUT_FILE \
                            -o $OUTPUT_FILE \
                            -d $DEVICE \
                            -nireq $NUM_REQS \
                            --labels $LABEL_FILE

# Run the output video annotator code
SCALE_FRAME_RATE=1    # scale number or output frames to input frames
SCALE_RESOLUTION=0.5  # scale output frame resolution 
python3 object_detection_annotate.py -i $INPUT_FILE \
                                     -o $OUTPUT_FILE \
                                     -f $SCALE_FRAME_RATE \
                                     -s $SCALE_RESOLUTION