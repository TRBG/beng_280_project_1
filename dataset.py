import os
from glob import glob

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, scans_list, scans_dir, sv_bp_dir, scan_ext, sv_bp_ext, num_of_views, transform=None):
        """
        Args:
            scans_list (list): scans list (file names without extension).
            scans_dir: scans file directory.
            sv_bp_dir: single view projection file directory.
            scan_ext (str): scan file extension.
            sv_bp_ext (str): single view projection file extension.
            transform (Compose, optional): Compose transforms of transformations.
        
        Important Note:
            The directory structure of the dataset shouyld be organized as follows:
            data_dir
                 └── <dataset name>
                        ├── scans
                        |   ├── 00001.png
                        │   ├── 00002.png
                        │   ├── 00003.png
                        │   ├── ...
                        |
                        └── sv_bp
                        	├── 00001_01.png
                            ├── 00001_02.png
                            ├── 00001_03.png
                            ├──	00001_04.png
                            ├── ...
                            
                            ├── 00002_01.png
                            ├── 00002_02.png
                            ├── 00002_03.png
                            ├── 00002_04.png
                            ├── ...
                			...
        """
        self.scans_list = scans_list
        self.scans_dir = scans_dir
        self.sv_bp_dir = sv_bp_dir
        self.scan_ext = scan_ext
        self.sv_bp_ext = sv_bp_ext
        self.transform = transform
        self.num_of_views = num_of_views

    def __len__(self):
        return len(self.scans_list)

    def __getitem__(self, idx):
        
        scan_id = self.scans_list[idx]
        scan = cv2.imread(os.path.join(self.scans_dir, scan_id + self.scan_ext), cv2.IMREAD_GRAYSCALE)[..., None]
        
        sv_bp = []                          
        for i in range(self.num_of_views):
            number_str = str(i+1)
            zero_filled_number = number_str.zfill(3)
            
            sv_bp.append(cv2.imread(os.path.join(self.sv_bp_dir, scan_id + '_' +
                        zero_filled_number + self.sv_bp_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        sv_bp = np.dstack(sv_bp)

        scan = scan.astype('float32')
        scan = scan.transpose(2, 0, 1)
        sv_bp = sv_bp.astype('float32')
        sv_bp = sv_bp.transpose(2, 0, 1)
        
        return sv_bp, scan, {'scan_id': scan_id}
