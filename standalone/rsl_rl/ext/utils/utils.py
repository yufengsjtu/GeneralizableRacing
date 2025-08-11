import torch
import csv
import atexit

class InfoLogger:
    _instance = None

    def __new__(
        cls, 
        file_name="info_log.csv", 
        data_type=["throttle", "cmd_wx", "cmd_wy", "cmd_wz", "roll", "pitch", "yaw", "wx", "wy", "wz", "ax", "ay", "az"]
        ):
        if cls._instance is None:
            cls._instance = super(InfoLogger, cls).__new__(cls)
            cls._instance._initialize(file_name, data_type)
        return cls._instance
    
    def _initialize(self, file_name, data_type:list[str]):
        self.file_name = file_name
        self.file = open(self.file_name, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        self.data_type = data_type
        # write header
        self.file.seek(0)
        if self.file.tell() == 0:
            self.writer.writerow(data_type)
        
        atexit.register(self._shutdown)

    def log_frame(self, frame_data:dict[str, float]):
        try:
            # convert to numpy
            frame_data = [frame_data.get(key, 'unknown') for key in self.data_type]

            # write to file
            self.writer.writerow(frame_data)
            self.file.flush()
        except Exception as e:
            print(f"[ERROR]: {e}")
            print("[INFO]: Failed to log frame data.")

    def _shutdown(self):
        if not self.file.closed:
            self.file.close()
            print(f"[INFO]: {self.file_name} closed.")


import numpy as np
import torch
import random
import os
import warp as wp
def set_seed(seed, deterministic=True):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
        # will cause training slow down[only for reproduciblity]
        if deterministic:
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            torch.use_deterministic_algorithms(True)
    wp.rand_init(seed)