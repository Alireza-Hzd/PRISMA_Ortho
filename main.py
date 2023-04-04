# This is a Python script for automatically correct the PRISMA RPC using GCPs from Sentinel-2
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    data_folder = Path("/Volumes/Samsung_T5/Satellite_Imagery/PRISMA")
    file_name = "PRS_L2C_STD_20200420104916_20200420104920_0001.he5"

    f = h5py.File(data_folder / file_name, 'r')

    print("Image Acquisition Start Time:", f.attrs.get('Product_StartTime'))
    print("UL Corner:", f.attrs.get('Product_ULcorner_lat'), f.attrs.get('Product_ULcorner_long'))
    print("LR Corner:", f.attrs.get('Product_LRcorner_lat'), f.attrs.get('Product_LRcorner_long'))

    pan = f['HDFEOS']['SWATHS']['PRS_L2C_PCO']['Data Fields']['Cube']

    plt.imshow(pan, vmin=np.nanpercentile(pan, 2), vmax=np.nanpercentile(pan, 98))
    plt.title('Pancromatic')
    plt.colorbar()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
