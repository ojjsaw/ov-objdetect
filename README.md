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

python3 object_detection.py -m models/mobilenet-ssd/FP16/mobilenet-ssd.xml \
                            -i data/reference-sample-data/object-detection-python/cars_1900.mp4 \
                            -o results/ \
                            -d CPU \
                            -nireq 4 \
                            --labels data/reference-sample-data/object-detection-python/labels.txt
```

```
openvino-dev[tensorflow2,mxnet,caffe,onnx,pytorch,kaldi]
```