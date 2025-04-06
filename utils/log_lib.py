import os
import logging

### *--- design for advanced log record ---* ###
class Logging:

    def __init__(self,
                 home_dir='.',
                 log_name=None,
                 log_level='INFO',
                 **kwargs) -> None:

        self.home_dir = home_dir
        self.log_name = kwargs.get('log_name', log_name)
        self.log_level = kwargs.get('log_level', log_level)

        log_path = os.path.join(self.home_dir,
                                self.log_name) if self.log_name else None

        logging.basicConfig(filename=log_path,
                            format='[%(levelname)s] %(asctime)s : %(message)s',
                            filemode='a',
                            level=logging.getLevelName(self.log_level),
                            datefmt='%Y-%m-%d %H:%M:%S')
