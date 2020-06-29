import logging
import logging.handlers
import datetime
import os


class _Logger(logging.Logger):
    def __init__(self, script_name):
        super().__init__('logger')

        if not os.path.exists('log'):
            os.mkdir('log')

        formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
        stream_handler = logging.StreamHandler()
        log_file_name = 'log/' + script_name + str(datetime.datetime.now()).replace(':', '-') + '.txt'
        file_handler = logging.FileHandler(filename=log_file_name)

        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self.addHandler(stream_handler)
        self.addHandler(file_handler)


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Log(_Logger, metaclass=_Singleton):
    pass
