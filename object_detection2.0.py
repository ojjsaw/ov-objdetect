"""
 Copyright (C) 2018-2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
# pylint: disable=E1101

import logging as log
from argparse import ArgumentParser
import sys
import time
import numpy as np
from openvino.runtime import AsyncInferQueue, Core, CompiledModel
import cv2
import os
from qarpo.demoutils import progressUpdate

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

results = {}

def build_argparser():
    """Input Arguments"""
    parser = ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Path to an .xml file with a trained model.',
                        required=True,
                        type=str)
    parser.add_argument('-i', '--input',
                        help='Path to input video file.',
                        required=True,
                        type=str)
    parser.add_argument('-d', '--device',
                        help='Specify the target infer device to; CPU, GPU, MYRIAD, or HDDL.'
                             '(CPU by default).',
                        default='CPU',
                        type=str)
    parser.add_argument('-l', '--labels',
                        help='Labels mapping file.',
                        default=None,
                        type=str)
    parser.add_argument('-pt', '--prob_threshold',
                        help='Probability threshold for detection filtering.',
                        default=0.5,
                        type=float)
    parser.add_argument('-o', '--output_dir',
                        help='Location to store the results of the processing',
                        default=None,
                        required=True,
                        type=str)
    return parser

def processBoxes(frame_count, res, labels_map, prob_threshold, initial_w, initial_h, result_file):
    for obj in res:
        dims = ""
        # Draw only objects when probability more than specified threshold
        if obj[2] > prob_threshold:
            class_id = int(obj[1])
            det_label = labels_map[class_id] if labels_map else "class="+str(class_id)
            dims = "{frame_id} {xmin} {ymin} {xmax} {ymax} {class_id} {det_label} {est} {time} \n".format(
               frame_id=frame_count,
               xmin=int(obj[3] * initial_w),
               ymin=int(obj[4] * initial_h),
               xmax=int(obj[5] * initial_w),
               ymax=int(obj[6] * initial_h),
               class_id=class_id, det_label=det_label,
               est=round(obj[2]*100, 1),
               time='N/A')
            result_file.write(dims)

def callback(request, frame_id):
    """Callback for each infer request in the queue """
    print(frame_id)
    # Copy the data from output tensors to numpy array and process it
    #results_copy = {output: data[:] for output, data in request.results.items()}
    #results.append(process_results(results_copy, frame_id))

def main():
    """EntryPoint for the program."""
    args = build_argparser().parse_args()

    # Create OpenVINO™ Runtime Core
    core = Core()

    # Compile the model
    compiled_model = core.compile_model(model=args.model, device_name=args.device)
    if isinstance(compiled_model, CompiledModel):
        log.info('Successfully Compiled model (%s) for (%s) device', args.model, args.device)

    # create async queue with optimal number of infer requests
    infer_queue = AsyncInferQueue(compiled_model)
    infer_queue.set_callback(callback)

    # Get input and output nodes.
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Setup output file for the program
    #job_id = str(os.environ['PBS_JOBID']).split('.')[0]
    job_id = "123456"
    result_file = open(os.path.join(args.output_dir, job_id, 'output.txt'), "w")
    pre_infer_file = os.path.join(args.output_dir, job_id, 'pre_progress.txt')
    infer_file = os.path.join(args.output_dir, job_id, 'i_progress.txt')
    processed_vid = os.path.join(args.output_dir, job_id, 'processed_vid.bin')

    # Input layer: batch size(n), channels (c), height(h), width(w)
    n, c, h, w = input_layer.shape
    log.info('N:%d C:%d H:%d W:%d', n, c, h, w)

    # Preprocess video file
    cap = cv2.VideoCapture(args.input)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    CHUNKSIZE = n*c*w*h
    id_ = 0
    with open(processed_vid, 'w+b') as file:
        time_start = time.time()
        while cap.isOpened():
            ret, next_frame = cap.read()
            if not ret:
                break
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            bin_frame = bytearray(in_frame)
            file.write(bin_frame)
            id_ += 1
            if id_%10 == 0: 
                progressUpdate(pre_infer_file, time.time()-time_start, id_, video_len)
    cap.release()

    # read labels file
    if args.labels:
        with open(args.labels, 'r', encoding="utf-8") as file:
            labels_map = [x.strip() for x in file]
    else:
        labels_map = None

    #input_data = 'd'
    #total_frames = 100
    #for i in range(total_frames):
        # Wait for at least one available infer request and start asynchronous inference
     #   infer_queue.start_async(next(input_data), userdata=i)

    infer_time_start = time.time()
    frame_count = 0
    with open(processed_vid, "rb") as data:
        while frame_count < video_len:
            byte = data.read(CHUNKSIZE)
            if not byte == b"":
                deserialized_bytes = np.frombuffer(byte, dtype=np.uint8)
                in_frame = np.reshape(deserialized_bytes, newshape=(n, c, h, w))
                infer_queue.start_async(in_frame, userdata=frame_count)
                frame_count += 1

    infer_queue.wait_all()

    del compiled_model


if __name__ == '__main__':
    sys.exit(main() or 0)
