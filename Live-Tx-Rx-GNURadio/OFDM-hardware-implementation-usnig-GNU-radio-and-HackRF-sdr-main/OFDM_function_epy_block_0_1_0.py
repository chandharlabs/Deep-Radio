"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const"""

    def __init__(self, fft_len = 1, default_log = 0):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='PAPR Calculator',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.float32]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        #arr[]
        self.fft_len = fft_len
        self.memory_log = default_log
        
    def work(self, input_items, output_items):
        arr = input_items[0][:]
        length = self.fft_len 
        i = 1  
        PAPR_LOG = 0
        if arr.size == 4096:
            while i <= 4096/self.fft_len:
                temp = arr[(i-1)*length : (i*length)]
                PAPR_LOG =20 * (np.log10(np.max(np.abs(temp)) / np.sqrt(np.mean(np.abs(temp)**2))))
                self.memory_log = PAPR_LOG
                output_items[0][:] = PAPR_LOG
                i = i + 1
            arr = []    
        else:
            output_items[0][:] = self.memory_log          
        return len(output_items[0])