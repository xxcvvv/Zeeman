import os
import time
import onnx
import onnxruntime as ort
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils.log_lib import Logging
from utils.data_lib import MyDataset
from utils.output_lib import pack_nc

### *----------------------------------------* ###
### *---      Load the configuration      ---* ###
### *----------------------------------------* ###
parser = argparse.ArgumentParser()
parser.add_argument('-start',
                    type=str,
                    default='2023-11-20 00:00',
                    help='Format : YYYY-MM-DD HH:MM')
parser.add_argument('-step',
                    type=int,
                    default=120,
                    help='Timesteps forward to predict')
parser.add_argument('-exp_name', type=str, default='standard')
parser.add_argument('-file', type=str, help='Onnx file path for inference')
parser.add_argument('-device',
                    type=str,
                    default='gpu',
                    help='Device to use for inference')
args = parser.parse_args()

out_chans = 187
name_list = [
    'no2', 'o3', 'co', 'so2', 'nh3', 'nh4a_f', 'pan', 'so4a_f', 'no3a_f',
    'no3a_c', 'ec', 'pom', 'ppm', 'tnmvoc', 'tpm25', 'tpm10', 'tss'
]
Logging(log_level='INFO')

start_time = datetime.strptime(args.start, '%Y-%m-%d %H:%M')
time_range = pd.date_range(start_time,
                           start_time + timedelta(hours=args.step),
                           freq='1h')

output_dir = os.path.join('inference', args.exp_name,
                          start_time.strftime('%Y%m%d_%H%M'))
os.makedirs(output_dir, exist_ok=True)

### *---------------------------------------* ###
### *---     Initialize the Dataset      ---* ###
### *---------------------------------------* ###

# build the dataloader
dataset = MyDataset(time_range[0], time_range[-1])

### *--------------------------------------* ###
### *---      Initialize the Model      ---* ###
### *--------------------------------------* ###
logging.info(f"Loading {args.file}")

if not os.path.exists(args.file):
    raise FileNotFoundError(f"{args.file} not found!")

# initialize the model
model = onnx.load(args.file)

# Set the behavier of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = 1

# Set the behavier of cuda provider
cuda_provider_options = {
    'arena_extend_strategy': 'kSameAsRequested',
}

if args.device == 'cpu':
    provider = ['CPUExecutionProvider']
elif args.device == 'gpu':
    provider = [('CUDAExecutionProvider', cuda_provider_options)]
else:
    raise ValueError(f"Unknown device: {args.device}")

logging.info(f"Using {args.device}")

# Initialize onnxruntime session for Pangu-Weather Models
ort_session = ort.InferenceSession(args.file,
                                   sess_options=options,
                                   providers=provider)

logging.info('Model loaded!')

### *-------------------------------------* ###
### *---      Start the Inference      ---* ###
### *-------------------------------------* ###
# Start inference
input_conc = None
output_array = np.zeros([len(dataset) + 2, out_chans, 40, 50])
target_array = np.zeros([len(dataset) + 2, out_chans, 40, 50])
logging.info('Inference started...')
for step in range(len(dataset)):

    # load data
    _input, boundary, target = dataset.load(step)
    conc3d, meteo3d, meteo2d, emis2d = _input

    # store the initial input data
    if step == 0:
        output_array[:2] = conc3d[0, :].reshape(2, out_chans, 40, 50)
        target_array[:2] = conc3d[0, :].reshape(2, out_chans, 40, 50)

    if step > 0:
        conc3d = conc3d_input

    # inference
    time_start = time.perf_counter()
    output = ort_session.run(
        None, {
            'conc3d': conc3d,
            'meteo3d': meteo3d,
            'meteo2d': meteo2d,
            'emis2d': emis2d,
            'boundary': boundary
        })[0]
    logging.info(
        f"inference cost {(time.perf_counter() - time_start)*1000:.2f} ms")
    logging.debug(f"mean of output : {output.mean()}, "
                  f"shape of output : {output.shape}")

    # store the data for next step
    input1 = conc3d[0, 1, :, :, :].reshape(1, out_chans, 40, 50)
    input2 = output.reshape(1, out_chans, 40, 50)
    logging.debug(
        f"mean of input1 : {input1.mean()}, mean of input 2 : {input2.mean()}")
    logging.debug(f"{input1.shape} | {input2.shape}")
    conc3d_input = np.concatenate([input1, input2])
    conc3d_input = conc3d_input.reshape(1, 2, out_chans, 40, 50)

    # calcultate the error
    mae = np.abs(output - target.reshape(output.shape)).mean()
    rmse = np.sqrt(np.mean((output - target.reshape(output.shape))**2))
    logging.info(f"step - {step} mae : {mae:.3f}, rmse : {rmse:.3f}")
    logging.debug(
        f"output | target : {input2.mean():.3f} | {target.mean():3f}")

    # store the output
    output_array[step + 2] = output.reshape(out_chans, 40, 50)
    target_array[step + 2] = target.reshape(out_chans, 40, 50)

### *--- Save the results ---* ###
pack_nc(f"{output_dir}/inference.nc", output_array, name_list, time_range)
pack_nc(f"{output_dir}/target.nc", target_array, name_list, time_range)
