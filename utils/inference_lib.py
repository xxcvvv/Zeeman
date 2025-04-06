import os
import time
import onnx
import onnxruntime as ort
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils.log_lib import Logging
from utils.data_lib import MyDataset
from utils.output_lib import pack_nc


class Zeeman:

    ### *--- Load the configuration ---* ###
    def __init__(self,
                 file: str,
                 device: str = 'gpu',
                 log_level: str = 'INFO') -> None:

        Logging(log_level=log_level)
        self.out_chans = 187
        self.name_list = [
            'no2', 'o3', 'co', 'so2', 'nh3', 'nh4a_f', 'pan', 'so4a_f',
            'no3a_f', 'no3a_c', 'ec', 'pom', 'ppm', 'tnmvoc', 'tpm25', 'tpm10',
            'tss'
        ]
        self.file = file
        self.device = device

        self._load_model()

    def _load_model(self, ):
        ### *--------------------------------------* ###
        ### *---      Initialize the Model      ---* ###
        ### *--------------------------------------* ###
        logging.info(f"Loading {self.file}")

        if not os.path.exists(self.file):
            raise FileNotFoundError(f"{self.file} not found!")

        # initialize the model
        onnx.load(self.file)

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

        if self.device == 'cpu':
            provider = ['CPUExecutionProvider']
        elif self.device == 'gpu':
            provider = [('CUDAExecutionProvider', cuda_provider_options)]
        else:
            raise ValueError(f"Unknown device: {self.device}")

        logging.info(f"Using {self.device}")

        # Initialize onnxruntime session for Pangu-Weather Models
        self.ort_session = ort.InferenceSession(self.file,
                                                sess_options=options,
                                                providers=provider)

        logging.info('Model loaded!')

    def inference(self,
                  start: str,
                  step: int,
                  exp_id: str = 'standard') -> None:
        '''
        start : str, start time of the inference, format : YYYY-MM-DD HH:MM
        step : int, number of steps to predict 
        '''

        logging.info(f"\n*-------------------------------\n"
                     f"*- ID: {exp_id}\n"
                     f"*-------------------------------")

        # build the dataset
        start_time = datetime.strptime(start, '%Y-%m-%d %H:%M')
        time_range = pd.date_range(start_time,
                                   start_time + timedelta(hours=step),
                                   freq='1h')
        dataset = MyDataset(time_range[0], time_range[-1])

        # Start inference
        output_array = np.zeros([len(dataset) + 2, self.out_chans, 40, 50])
        target_array = np.zeros([len(dataset) + 2, self.out_chans, 40, 50])
        logging.info('Inference started...')
        for step in range(len(dataset)):

            # read data
            _input, boundary, target = dataset.load(step)
            conc3d, meteo3d, meteo2d, emis2d = _input

            # store the initial input data
            if step == 0:
                output_array[:2] = conc3d[0, :].reshape(
                    2, self.out_chans, 40, 50)
                target_array[:2] = conc3d[0, :].reshape(
                    2, self.out_chans, 40, 50)

            if step > 0:
                conc3d = conc3d_input

            # inference
            time_start = time.perf_counter()
            output = self.ort_session.run(
                None, {
                    'conc3d': conc3d,
                    'meteo3d': meteo3d,
                    'meteo2d': meteo2d,
                    'emis2d': emis2d,
                    'boundary': boundary
                })[0]
            time_end = time.perf_counter()
            logging.debug(f"mean of output : {output.mean()}, "
                          f"shape of output : {output.shape}")

            # store the data for next step
            input1 = conc3d[0, 1, :, :, :].reshape(1, self.out_chans, 40, 50)
            input2 = output.reshape(1, self.out_chans, 40, 50)
            logging.debug(
                f"mean of input1 : {input1.mean()}, mean of input 2 : {input2.mean()}"
            )
            logging.debug(f"{input1.shape} | {input2.shape}")
            conc3d_input = np.concatenate([input1, input2])
            conc3d_input = conc3d_input.reshape(1, 2, self.out_chans, 40, 50)

            logging.debug(
                f"output | target : {input2.mean():.3f} | {target.mean():3f}")

            # calcultate the error
            mae = np.abs(output - target.reshape(output.shape)).mean()
            rmse = np.sqrt(np.mean((output - target.reshape(output.shape))**2))
            logging.info(f"step - {step} mae : {mae:.3f}, rmse : {rmse:.3f}")

            # store the output
            output_array[step + 2] = output.reshape(self.out_chans, 40, 50)
            target_array[step + 2] = target.reshape(self.out_chans, 40, 50)

        # pack the outputs and targets
        output_dir = os.path.join('inference', exp_id,
                                  start_time.strftime('%Y%m%d_%H%M'))
        os.makedirs(output_dir, exist_ok=True)
        self._pack_output(output_dir, output_array, target_array, time_range)

    def _pack_output(
        self,
        output_dir: str,
        output: np.ndarray,
        target: np.ndarray,
        time_range: pd.DatetimeIndex,
    ):
        ### *--- Save the results ---* ###
        pack_nc(f"{output_dir}/inference.nc", output, self.name_list,
                time_range)
        pack_nc(f"{output_dir}/target.nc", target, self.name_list, time_range)


if __name__ == '__main__':

    zeeman = Zeeman('model/Zeeman.onnx')
    zeeman.inference('2023-01-01 00:00', 120)
