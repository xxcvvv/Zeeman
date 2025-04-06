import logging
import logging
import numpy as np
import netCDF4 as nc
from pytz import utc
from datetime import datetime


### *--- Designed for nc file creation ---* ###
class NcProduct:

    def __init__(self,
                 file_path='test.nc',
                 mode='w',
                 format='NETCDF4',
                 **kwargs):
        """
        file_path: the path of the nc file
        mode: the mode of the nc file
        format: the format of the nc file
        """
        self.file_path = self.ensure_nc_extension(file_path)
        self.nc_file = nc.Dataset(self.file_path, mode, format=format)
        self.init_file(mode, **kwargs)

    @staticmethod
    def ensure_nc_extension(file_path: str) -> str:
        return file_path if file_path.endswith('.nc') else f'{file_path}.nc'

    ### *--- initiate the nc file ---* ###
    def init_file(self, mode: str, **kwargs) -> None:

        if mode == 'w':
            self.set_creation_time()
            self.set_attr_global(**kwargs)

    # set the file creation time
    def set_creation_time(self) -> None:
        self.set_attr_global(Creation_time=datetime.now().astimezone(
            utc).strftime('%Y-%m-%d %H:%M:%S UTC'))

    ### *--- Define the Nc File  ---* ###
    # dimension
    def define_dimension(self, **dims) -> None:

        for name, size in dims.items():
            self.nc_file.createDimension(name, size=size)

    # variables
    def define_variable(self, **kwargs) -> None:
        # kwargs:dict{str:tuple}

        for var_name, (var_type, dimensions) in kwargs.items():
            self.nc_file.createVariable(var_name, var_type, dimensions)

    # variable by dict
    def define_variable_dict(self, var_dict: dict):

        for i_key in var_dict.keys():
            self.nc_file.createVariable(i_key,
                                        var_dict[i_key][0],
                                        dimensions=var_dict[i_key][1])

    ### *--- Set Attributions ---* ###
    # global attributions
    def set_attr_global(self, **kwargs) -> None:
        for key in kwargs.keys():
            self.nc_file.setncattr(key, kwargs[key])

    # variable attributions
    def set_attr(self, **kwargs) -> None:
        for var in kwargs.keys():
            for attr_name in kwargs[var].keys():
                self.nc_file.variables[var].setncattr(attr_name,
                                                      kwargs[var][attr_name])

    ### *--- add the data ---* ###
    def add_data(self, count=1e9, **kwargs) -> None:

        for key in kwargs.keys():

            ### determine if the variables contain a unlimited dimendion ###
            ### Attention： the first dimension must be time if there is unlimited dim ###
            unlimit_flag = False
            time_dim = str(self.nc_file[key].dimensions[0])
            if self.nc_file.dimensions[time_dim].isunlimited():
                unlimit_flag = True

            ### *--- time dimension ---* ###
            if unlimit_flag:

                time_dim_size = self.nc_file.dimensions[time_dim].size
                # assign the dim to add the data
                if not count == 1e9:
                    self.nc_file[key][count] = kwargs[key]
                # append the data after the last time dim
                else:
                    self.nc_file[key][time_dim_size] = kwargs[key]

            ### *--- not time dimension ---* ###
            else:

                self.nc_file[key][:] = kwargs[key]

    ### *--- close the nc file ---* ###
    def close(self) -> None:

        self.set_attr_global(
            history='Last modified at %s UTC' %
            datetime.now().astimezone(utc).strftime('%Y-%m-%d %H:%M:%S'))
        self.nc_file.close()


def pack_nc(file_path: str,
            data: np.ndarray,
            name_list: list,
            time_list: list[datetime],
            lev: int = 11,
            mode: str = 'w',
            **kwargs) -> None:

    nc_file = NcProduct(file_path=file_path, mode=mode, **kwargs)
    nc_file.set_attr_global(Model='Zeeman', version='1')
    nc_file.define_dimension(lon=50, lat=40, lev=11, time=None)
    nc_file.define_variable(lon=['f4', 'lon'],
                            lat=['f4', 'lat'],
                            lev=['f4', 'lev'],
                            time=['S19', 'time'])

    for i_name in range(len(name_list)):
        nc_file.define_variable(
            **{name_list[i_name]: ['f4', ('time', 'lev', 'lat', 'lon')]})

    for i_time in range(len(time_list)):
        nc_file.add_data(time=time_list[i_time].strftime('%Y-%m-%d %H:%M:%S'))
        lev_count = 0
        for i_name in range(len(name_list)):
            nc_file.add_data(count=i_time,
                             **{
                                 name_list[i_name]:
                                 data[i_time, lev_count:lev_count + lev, :, :]
                             })
            lev_count += lev

    nc_file.close()
    logging.info(f"{file_path} is created")
