import os
import logging
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta


def is_same_day(time1: datetime, time2: datetime) -> bool:
    if time1.strftime('%Y%m%d') == time2.strftime('%Y%m%d'):
        return True
    else:
        return False


# check if any variable is missing in file
def check_vars(required_vars: list[str],
               existing_vars: list[str],
               file_name: str = '') -> bool:

    missing_var = []
    for var in required_vars:
        if not var in existing_vars:
            missing_var.append(var)
    if len(missing_var) > 0:
        logging.warning(f"<{missing_var}> is not in the file {file_name} !")
        return False
    else:
        return True


def reader(data_path: str,
           times: list[datetime],
           names: list[str],
           slice_list=slice(None)) -> np.ndarray:
    # open the file
    with nc.Dataset(data_path) as nc_obj:

        # check if any variable is missing in file
        if not check_vars(names, nc_obj.variables, data_path):
            return None
        else:
            # index the time
            time_idx = tuple(
                int(times[i_time].strftime('%H'))
                for i_time in range(len(times)))

            # get the data
            data = np.array(
                [[nc_obj.variables[name][time][slice_list] for name in names]
                 for time in time_idx])

    return data


def read_data(data_dir: str,
              file_tag: str,
              times: list[datetime],
              names: list[str],
              slice_list=slice(None)) -> np.ndarray:

    if not isinstance(times, list):
        raise ValueError('times should be a list of datetime objects.')

    if len(times) == 1:
        data = reader(os.path.join(
            data_dir, f"{file_tag}_{times[0].strftime('%Y%m%d')}.nc"),
                      times,
                      names,
                      slice_list=slice_list)
    else:
        if is_same_day(times[0], times[1]):
            data = reader(os.path.join(
                data_dir, f"{file_tag}_{times[0].strftime('%Y%m%d')}.nc"),
                          times,
                          names,
                          slice_list=slice_list)
        else:
            data = np.array([
                reader(os.path.join(
                    data_dir, f"{file_tag}_{times[0].strftime('%Y%m%d')}.nc"),
                       [times[0]],
                       names,
                       slice_list=slice_list),
                reader(os.path.join(
                    data_dir, f"{file_tag}_{times[1].strftime('%Y%m%d')}.nc"),
                       [times[1]],
                       names,
                       slice_list=slice_list)
            ])

    return data


class ConcData:

    def __init__(self, data_dir: str) -> None:

        self.data_dir = data_dir

    def get_data(self, time: list[datetime], names: list[str]) -> np.ndarray:
        """
        Get the concentration data from the netcdf file.
        """
        data = read_data(self.data_dir, 'LE_MT_conc-3d', time, names)

        if data is not None:
            data *= 10**9

        return data


class BoundaryData:

    def __init__(self, data_dir: str) -> None:

        self.data_dir = data_dir

    def get_data(self, time: list[datetime], names: list[str]) -> np.ndarray:
        """
        Get the concentration data from the netcdf file.
        """
        data = read_data(self.data_dir,
                         'LE_MT_conc-halo',
                         time,
                         names,
                         slice_list=(slice(-1), slice(None), slice(None)))

        if data is not None:
            data *= 10**9

        return data


class MeteoData:

    def __init__(self, data_dir: str) -> None:

        self.data_dir = data_dir

    def get_data_3d(self, time: list[datetime],
                    names: list[str]) -> np.ndarray:

        data = read_data(self.data_dir, 'LE_MT_meteo-3d', time, names)

        return data

    def get_data_2d(self, time: list[datetime],
                    names: list[str]) -> np.ndarray:

        data = read_data(self.data_dir, 'LE_MT_meteo-2d', time, names)

        return data


class EmisData:

    def __init__(self, data_dir: str) -> None:

        self.data_dir = data_dir

    def get_data(self, time: list[datetime], names: list[str]) -> np.ndarray:

        # prepare the necessary variables
        # time_in_name = time - timedelta(hours=1)

        data = read_data(self.data_dir,
                         'LE_MT_emis',
                         time,
                         names,
                         slice_list=(slice(1), slice(None), slice(None)))

        if data is not None:
            data *= 10**12

        return data


class MyDataset():

    def __init__(
            self,
            start: datetime,
            end: datetime,
            name_dict={
                'conc': [
                    'no2', 'o3', 'co', 'so2', 'nh3', 'nh4a_f', 'pan', 'so4a_f',
                    'no3a_f', 'no3a_c', 'ec', 'pom', 'ppm', 'tnmvoc', 'tpm25',
                    'tpm10', 'tss'
                ],
                'meteo_3d': ['t', 'rh', 'uv_u', 'uv_v', 'p'],
                'meteo_2d': ['blh', 'rain'],
                'emis_2d': [
                    'no2', 'no', 'co', 'form', 'ald', 'par', 'ole', 'eth',
                    'tol', 'xyl', 'so4a_f', 'so2', 'ch4', 'nh3', 'iso', 'terp',
                    'ec_f', 'ec_c', 'pom_f', 'pom_c', 'ppm_f', 'ppm_c',
                    'na_ff', 'na_f', 'na_ccc', 'na_cc', 'na_c'
                ]
            },
            interval: int = 1) -> None:

        # generate run time range
        input_times = pd.date_range(start, end, freq=f"{interval}h")

        logging.info(f"Length of dataset: {len(input_times) - 2 * interval}")
        logging.info(
            f"Input time range: {input_times[0]} to {input_times[-1]}")
        logging.info(f"Lead time: {interval}h")

        # initiate the data class
        cdata = ConcData('data/conc-3d')
        mdata = MeteoData('data/meteo')
        edata = EmisData('data/emis')
        bdata = BoundaryData('data/conc-halo')

        self.cdata_names = name_dict['conc']
        self.mdata_3d_names = name_dict['meteo_3d']
        self.mdata_2d_names = name_dict['meteo_2d']
        self.edata_names = name_dict['emis_2d']
        self.bdata_names = self.cdata_names
        self.target_names = self.cdata_names

        self.cdata = cdata
        self.mdata = mdata
        self.edata = edata
        self.bdata = bdata

        self.interval = interval
        self.input_times = input_times

    def __len__(self) -> int:
        return len(self.input_times)

    def load(self, step: int) -> list[np.ndarray]:

        ### *--- prepare config ---* ###
        input_time = self.input_times[step]
        input_time_before = input_time - timedelta(hours=self.interval)
        target_time = input_time + timedelta(hours=self.interval)

        # get the data
        cdata_3d = self.cdata.get_data([input_time_before, input_time],
                                       self.cdata_names)
        mdata_3d = self.mdata.get_data_3d([input_time_before, input_time],
                                          self.mdata_3d_names)
        edata_2d = self.edata.get_data([input_time_before, input_time],
                                       self.edata_names)
        mdata_2d = self.mdata.get_data_2d([input_time_before, input_time],
                                          self.mdata_2d_names)
        bdata_3d = self.bdata.get_data([input_time_before, input_time],
                                       self.bdata_names)

        cdata_3d_target = self.cdata.get_data([target_time], self.target_names)

        # decide if there are none value
        if any(data is None for data in [
                cdata_3d, mdata_3d, bdata_3d, edata_2d, mdata_2d,
                cdata_3d_target
        ]):
            return None

        # convert to prefered shape
        cdata_3d = cdata_3d.reshape(1, 2, -1, cdata_3d.shape[-2],
                                    cdata_3d.shape[-1]).astype(np.float32)
        mdata_3d = mdata_3d.reshape(1, 2, -1, mdata_3d.shape[-2],
                                    mdata_3d.shape[-1]).astype(np.float32)
        edata_2d = edata_2d.reshape(1, 2, -1, edata_2d.shape[-2],
                                    edata_2d.shape[-1]).astype(np.float32)
        mdata_2d = mdata_2d.reshape(1, 2, -1, mdata_2d.shape[-2],
                                    mdata_2d.shape[-1]).astype(np.float32)
        
        bdata_3d = bdata_3d.reshape(2, -1, bdata_3d.shape[-2],
                                    bdata_3d.shape[-1])
        bdata_3d = self.cut_boundary(bdata_3d)
        bdata_3d = bdata_3d.reshape(1, 2, -1, bdata_3d.shape[-2],
                                    bdata_3d.shape[-1]).astype(np.float32)

        target = cdata_3d_target.astype(np.float32)

        logging.debug(f"shape of cdata_3d: {cdata_3d.shape}")
        logging.debug(f"shape of bdata_3d: {bdata_3d.shape}")
        logging.debug(f"shape of mdata_2d: {mdata_2d.shape}")
        logging.debug(f"shape of edata_2d: {edata_2d.shape}")
        logging.debug(f"shape of mdata_3d: {mdata_3d.shape}")
        logging.debug(f"shape of target: {target.shape}")

        return [cdata_3d, mdata_3d, mdata_2d, edata_2d], bdata_3d, target

    @staticmethod
    def cut_boundary(boundary: np.ndarray) -> np.ndarray:
        # shape of input boundary includes 2 extra layers
        T, C, H, W = boundary.shape
        H, W = H - 2, W - 2
        _boundary = np.zeros([T, C, 2, H + W])
        _boundary[:, :, 0, :W] = boundary[:, :, 0, 1:-1]  # bottom
        _boundary[:, :, 1, :W] = boundary[:, :, -1, 1:-1]  # up
        _boundary[:, :, 0, W:] = boundary[:, :, 1:-1, 0]  # left
        _boundary[:, :, 1, W:] = boundary[:, :, 1:-1, -1]  # right
        return _boundary
