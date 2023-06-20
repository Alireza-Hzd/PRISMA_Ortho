import h5py
import os
import rasterio
import numpy as np
import pandas as pd
import rpcsat

from sklearn.model_selection import KFold
from pathlib import Path
from osgeo import gdal
from datetime import datetime


class PrismaData:
    def __init__(self, filepath):
        self.filepath = Path(filepath)

        with h5py.File(self.filepath, 'r') as h5file:

            self.minLat, self.maxLat = float(h5file.attrs.get('Product_LRcorner_lat')), float(h5file.attrs.get('Product_ULcorner_lat'))
            self.minLon, self.maxLon = float(h5file.attrs.get('Product_LRcorner_long')), float(h5file.attrs.get('Product_ULcorner_long'))

            self.RPC_Hyp = rpcsat.RPCmodel()
            self.RPC_Hyp.read_from_PRISMA_h5(self.filepath)

            self.RPC_Pan = rpcsat.RPCmodel()
            self.RPC_Pan.read_from_PRISMA_h5(self.filepath, panchromatic=True)

            self.startTime = h5file.attrs.get('Product_StartTime')
            self.date = datetime.strptime(str(self.startTime).split("T")[0].split("'")[-1], "%Y-%m-%d")

            self.vnir_central_wavelengths = h5file.attrs['List_Cw_Vnir']
            self.vnir_bands_amplitude = h5file.attrs['List_Fwhm_Vnir']

            self.swir_central_wavelengths = h5file.attrs['List_Cw_Swir']
            self.swir_bands_amplitude = h5file.attrs['List_Fwhm_Swir']

            self.L2ScaleVnirMin = h5file.attrs['L2ScaleVnirMin']
            self.L2ScaleVnirMax = h5file.attrs['L2ScaleVnirMax']

            self.L2ScaleSwirMin = h5file.attrs['L2ScaleSwirMin']
            self.L2ScaleSwirMax = h5file.attrs['L2ScaleSwirMax']


        self.hyp_cube = None
        self.pan_band = None

    def read_hyp_cube(self):
        with h5py.File(self.filepath, 'r') as h5f:
            # read SWIR and VNIR cube contents

            SWIRcube = h5f['HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/SWIR_Cube'][()]
            VNIRcube = h5f['HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/VNIR_Cube'][()]



            # read latitude and longitude contents
            lat = h5f['HDFEOS/SWATHS/PRS_L2C_HCO/Geolocation Fields/Latitude'][()]

            # checks SWIR, VNIR and latitude/longitude array shapes
            print("SWIRcube.shape", SWIRcube.shape)
            print("VNIRcube.shape", VNIRcube.shape)
            #print("lat.shape", lat.shape)

            # ADD the DN to reflectance conversion
            VNIRcube = self.L2ScaleVnirMin + (VNIRcube * (self.L2ScaleVnirMax-self.L2ScaleVnirMin))/65535

            SWIRcube = self.L2ScaleSwirMin + (SWIRcube * (self.L2ScaleSwirMax - self.L2ScaleSwirMin))/65535

            # create a list from VNIR and SWIR cube values
            listBand = []
            for band in range(0, VNIRcube.shape[1]):  # VNIRcube.shape[1] gives the number of bands (:66)
                for x in range(0, lat.shape[0]):  # lat.shape[0] gives the number of rows
                    element = VNIRcube[x][band]
                    listBand.append(element)
            for band1 in range(0, SWIRcube.shape[1]):  # SWIRcube.shape[1] gives the number of bands (:137)
                for x1 in range(0, lat.shape[0]):  # lat.shape[0] gives the number of rows
                    element = SWIRcube[x1][band1]
                    listBand.append(element)

            # convert list with values to a numpy array
            data = np.array(listBand, dtype=np.float16)

            # checks array shape
            print("data.shape", data.shape)

            # reshape numpy array with the right number of bands, rows and columns
            dataReshaped = data.reshape([VNIRcube.shape[1] + SWIRcube.shape[1], lat.shape[0], lat.shape[1]])
            print("reshaped data.shape", dataReshaped.shape)

            self.hyp_cube = dataReshaped

    #def get_spectral_index(self, indxb1, inxb2):
    #    return

    def get_hyp_cube(self):
        if self.hyp_cube is None:
            self.read_hyp_cube()
        if self.hyp_cube is None:
            raise ValueError("Data has not been read from the input file.")

        return self.hyp_cube

    def raw_data_export_to_geotiff(self, output_file):

        if self.hyp_cube is None:
            raise ValueError("Data has not been read from the input file.")

        rpc_dict = self.RPC_Hyp.to_geotiff_dict()

        with rasterio.open(
                output_file, "w",
                driver="GTiff",
                height=self.hyp_cube.shape[1],
                width=self.hyp_cube.shape[2],
                count=self.hyp_cube.shape[0],
                dtype=self.hyp_cube.dtype,
                crs=4326
        ) as dst:
            dst.update_tags(ns="RPC", **rpc_dict)
            dst.write(self.hyp_cube)


    def orthorectify_hyp_cube(self, output_file_path, dem_file_path, GCPs_data=None):
        if self.hyp_cube is None:
            self.read_hyp_cube()

        if self.hyp_cube is None:
            raise ValueError("Data has not been read from the input file.")

        gcp_flag = ""
        if GCPs_data is not None:
            self.RPC_Hyp.GCP_assessment(GCPs_data)
            N_GCP = GCPs_data.shape[0]

            print("")
            print("--- Leave one out (K-fold) validation ---")

            kf = KFold(n_splits=N_GCP)
            kf.get_n_splits(GCPs_data)
            results_df = pd.DataFrame()

            for train_index, test_index in kf.split(GCPs_data):
                X_train, X_test = GCPs_data.iloc[train_index], GCPs_data.iloc[test_index]
                rpc_img_n = self.RPC_Hyp
                rpc_img_n.GCP_refinement(X_train.reset_index(), verbose=False)
                Row_mean, Row_std, Col_mean, Col_std = rpc_img_n.GCP_assessment(X_test.reset_index(), verbose=False)
                X_test_copy = X_test.copy()
                X_test_copy['D_Col'] = [Col_mean]
                X_test_copy['D_Row'] = [Row_mean]
                X_test_copy['Mod'] = [np.sqrt(Row_mean * Row_mean + Col_mean * Col_mean)]
                results_df = pd.concat([results_df, X_test_copy], ignore_index=True)

            print(results_df.head(30))
            print("")
            print("--- Leave One Out Residual errors N_GCP:", N_GCP, " ----")
            print("Row_mean:", results_df[['D_Row']].mean(axis=0).to_numpy()[0], "Row_std:",
                  results_df[['D_Row']].std(axis=0).to_numpy()[0])
            print("Col_mean:", results_df[['D_Col']].mean(axis=0).to_numpy()[0], "Col_std:",
                  results_df[['D_Col']].std(axis=0).to_numpy()[0])

            self.RPC_Hyp.GCP_refinement(GCPs_data)
            gcp_flag = "gcp"

        output_file_path = Path(output_file_path)

        tmp_file = output_file_path.parent / (output_file_path.stem + "_raw.tif")

        self.raw_data_export_to_geotiff(tmp_file.as_posix())

        gdal_option_string = "-r near -rpc -to 'RPC_DEM=" + dem_file_path + "' -of GTiff"

        ds_input = gdal.OpenEx(tmp_file.as_posix(), gdal.OF_RASTER)

        ds = gdal.Warp(output_file_path.as_posix(), ds_input,
                       options=gdal_option_string)
        del ds
        del ds_input

        os.remove(tmp_file)

        print("Orthorectified hypercube save to: ", output_file_path.as_posix())


    def refine_rpc(self):
        if self.data is None:
            raise ValueError("Data has not been read from the input file.")

        # Perform RPC refinement
        # ...
        # Add your code here

        # Update metadata
        # ...
        # Add your code here