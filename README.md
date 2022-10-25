# ov-objdetect

### Download model
```
omz_downloader --name mobilenet-ssd -o raw_models
```

### Create FP16 IR files
```
mo \
--input_model raw_models/public/mobilenet-ssd/mobilenet-ssd.caffemodel \
--data_type FP16 \
--output_dir models/mobilenet-ssd/FP16 \
--scale 256 \
--mean_values [127,127,127] 
```

### Create FP32 IR files
```
mo \
--input_model raw_models/public/mobilenet-ssd/mobilenet-ssd.caffemodel \
--data_type FP32 \
--output_dir models/mobilenet-ssd/FP32 \
--scale 256 \
--mean_values [127,127,127] 
```

```
mkdir -p results/123456

# Run the object detection code
python3 object_detection.py -m models/mobilenet-ssd/FP16/mobilenet-ssd.xml \
                            -i data/reference-sample-data/object-detection-python/cars_1900.mp4 \
                            -o results/ \
                            -d CPU \
                            -nireq 4 \
                            --labels data/reference-sample-data/object-detection-python/labels.txt
```

```
python3 object_detection2.0.py -m models/mobilenet-ssd/FP16/mobilenet-ssd.xml \
                            -i data/reference-sample-data/object-detection-python/cars_1900.mp4 \
                            -o results/ \
                            -d CPU \
                            --labels data/reference-sample-data/object-detection-python/labels.txt
```

```
# Run the output video annotator code
SCALE_FRAME_RATE=1    # scale number or output frames to input frames
SCALE_RESOLUTION=0.5  # scale output frame resolution 
python3 object_detection_annotate.py -i $INPUT_FILE \
                                     -o $OUTPUT_FILE \
                                     -f $SCALE_FRAME_RATE \
                                     -s $SCALE_RESOLUTION
```

```

python3 -m venv openvino_env && source openvino_env/bin/activate && python -m pip install --upgrade pip


openvino-dev[tensorflow2,mxnet,caffe,onnx,pytorch,kaldi]

pip3 install -r requirements-cert.txt
```