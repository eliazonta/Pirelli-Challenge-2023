import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_blobs

class PressData:
    def __init__(self, n_rows):
        self.n_rows = n_rows
        self.barcodes = self.create_barcode(self.n_rows)
        self.ipcodes = self.create_ipcode(self.n_rows)
        self.arms = self.create_arms(self.n_rows)
        
    def create_barcode(self):
        bars = []
        for i in range(self.n_rows):
            bars.append(int(random.randrange(955555500, 955555600)))
        return bars
    
    def create_ipcode(self):
        ips = []
        for i in range(self.n_rows):
            ips.append(int(random.randrange(15000,15500)))
        return ips
    
    def create_arms(self):
        arms = []
        for i in range(self.n_rows):
            tmp = random.randint(0,1)
            if(tmp == 0):
                tmp = 'R'
            else:
                tmp = 'L'
            arms.append(tmp)
        return arms
    
    def create_timestamp(self):
        # https://datascience.stackexchange.com/questions/54686/generating-artificial-time-series-data
        pass
    
        

class OldPress(PressData):
    def __init__(self, n_rows) -> None:
        super.__init__(n_rows)
        self.events = [
            "LO_LOADER_IN_PRESS",
            "LO_BLADDER_PRESHAPING",
            "LO_BLADDER_VACUUM",
            "LO_LOADER_DOWN",
            "LO_TCR_DOWN",
            "LO_PRESHAPING",
            "LO_LOADER_RELEASE_TIRE",
            "LO_LOADER_UP",
            "LO_LOADER_OUT",
            
            "CL_UNCLOCK_PRESS",
            "CL_PRESS_DOWN",
            "CL_STOP_PAUSE",
            "CL_PRESS_DOWN_2",
            "CL_STOP_PAUSE_2",
            "CL_PRESS_DOWN_3",
            "CL_PRESS_LOCK",
            "CL_SQUEEZE_ON",
            
            "CURING_ON",
            "CURING_OFF",
            
            "OP_SQUEEZE_OFF",
            "OP_UNCLOCK_PRESS",
            "OP_PRESS_MOVEMENT_UP",
            "OP_PRESS_STOP_PAUSE_1",
            "OP_PRESS_MOVEMENT_UP_1",
            "OP_PRESS_STOP_PAUSE_2",
            "OP_PRESS_MOVEMENT_UP_2",
            "OP_PRESS_STOP_PAUSE_3",
            "OP_PRESS_MOVEMENT_UP_3",
            "OP_PRESS_LOCKED",
            
            "UN_LMR_UP",
            "UN_FORK_IN",
            "UN_VACUUM",
            "UN_TCR_UP_LMR_DOWN",
            "UN_TCR_DOWN"
            "UN_FORK_OUT"            
        ]
    
    def create_machine_codes(self):
        mach = []
        for i in range(self.n_rows):
            mach.append(str(int(random.randrange(1200.1250))))
        return mach
    
    

class NewPress(PressData):
    def __init__(self, n_rows) -> None:
        super.__init__(n_rows)
        self.events = [
            "LO_LOADER_IN_PRESS",
            "LO_BLADDER_PRESHAPING",
            "LO_BLADDER_VACUUM",
            "LO_LOADER_DOWN",            
            "LO_PRESHAPING",
            "LO_TCR_DOWN",
            "LO_LOADER_RELEASE_TIRE",
            "LO_LOADER_UP",
            "LO_LOADER_OUT",
            
            "CL_UNCLOCK_PRESS",
            "CL_PRESS_DOWN",
            "CL_STOP_PAUSE",
            "CL_PRESS_DOWN_2",
            "CL_STOP_PAUSE_2",
            "CL_PRESS_DOWN_3",
            "CL_PRESS_LOCK",
            "CL_SQUEEZE_ON",
            
            "CURING_ON",
            "CURING_OFF",
            
            "OP_SQUEEZE_OFF",
            "OP_UNCLOCK_PRESS",
            "OP_PRESS_MOVEMENT_UP",
            "OP_PRESS_STOP_PAUSE_1",
            "OP_PRESS_MOVEMENT_UP_1",
            "OP_PRESS_STOP_PAUSE_2",
            "OP_PRESS_MOVEMENT_UP_2",
            "OP_PRESS_STOP_PAUSE_3",
            "OP_PRESS_MOVEMENT_UP_3",
            "OP_PRESS_LOCKED",
            
            "UN_SWING_IN_ARMS",
            "UN_TCR_UP",
            "UN_UNLOADER_DOWN",
            "UN_CLOSE_ARMS",
            "UN_OPEN_ARMS",
            "UN_TCR_UP_NO_SUV",
            "UN_UNLOADER_UP",
            "UN_UNLOADER_OUT"            
        ]
        
    def create_machine_codes(self):
        mach = []
        for i in range(self.n_rows):
            mach.append("C"+ str(int(random.randrange(1200.1250))))
        return mach
        
class Columns:
    barcode = "ddc_barcode" # 10 digits int code
    ipcode = "ddc_ipcode" # 5 digits int code
    machine = "ddc_mch_code" # 4 digits int code for ol machines, C+4 digits for new machines
    machine_side = "ddc_mch_side" # 'L' or 'R' char
    event = "ddc_ev_subcode" # string description of the event
    event_timestamp = "ddc_ev_timestamp" # timeseries tirmstamp of the event
    
class DataGenerator:
    pass
    
