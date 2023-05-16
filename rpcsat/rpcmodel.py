import numpy as np
import pandas as pd
#import statsmodels.api as sm
import numpy.linalg as la
import matplotlib.pyplot as plt
#import pygeodesy as geo
import xml.etree.ElementTree as et
import h5py

class RPCmodel():

    def __init__(self):

        self.LINE_Num_coeff = []
        self.LINE_Den_coeff = []
        self.SAMP_Num_coeff = []
        self.SAMP_Den_coeff = []
        self.LONG_SCALE = 0.0
        self.LONG_OFF = 0.0
        self.LAT_SCALE = 0.0
        self.LAT_OFF = 0.0
        self.HEIGHT_SCALE = 0.0
        self.HEIGHT_OFF = 0.0
        self.SAMP_SCALE = 0.0
        self.SAMP_OFF = 0.0
        self.LINE_SCALE = 0.0
        self.LINE_OFF = 0.0

    def read_from_ENVI(self, ENVI_file):
        with open(ENVI_file, encoding='utf-8', errors='ignore') as fp:
            for line in fp:
                dd = line.split("=")
                if dd[0].strip() == "Line Offset":
                    self.LINE_OFF_Text = dd[1]
                    self.LINE_OFF = np.double(dd[1])
                if dd[0].strip() == "Sample Offset":
                    self.SAMP_OFF = np.double(dd[1])
                if dd[0].strip() == "Latitude Offset":
                    self.LAT_OFF = np.double(dd[1])
                if dd[0].strip() == "Longitude Offset":
                    self.LONG_OFF = np.double(dd[1])
                if dd[0].strip() == "Height Offset":
                    self.HEIGHT_OFF = np.double(dd[1])

                if dd[0].strip() == "Line Scale":
                    self.LINE_SCALE = np.double(dd[1])
                if dd[0].strip() == "Sample Scale":
                    self.SAMP_SCALE = np.double(dd[1])
                if dd[0].strip() == "Latitude Scale":
                    self.LAT_SCALE = np.double(dd[1])
                if dd[0].strip() == "Longitude Scale":
                    self.LONG_SCALE = np.double(dd[1])
                if dd[0].strip() == "Height Scale":
                    self.HEIGHT_SCALE = np.double(dd[1])

                if "Line Numerator" in dd[0]:
                    self.LINE_Num_coeff.append(np.double(dd[1]))
                if "Line Denominator" in dd[0]:
                    self.LINE_Den_coeff.append(np.double(dd[1]))
                if "Sample Numerator" in dd[0]:
                    self.SAMP_Num_coeff.append(np.double(dd[1]))
                if "Sample Denominator" in dd[0]:
                    self.SAMP_Den_coeff.append(np.double(dd[1]))

    def read_from_xml(self, xml_file):

        sensor = "ICEYE"
        path =''
        xtree = et.parse(xml_file)
        xroot = xtree.getroot()

        child = xroot.find("RPC")

        # Read RPC coeff
        # Read LAT;LON;H Scale and Offset

        if sensor == "ICEYE":
            path = "./Metadata/RPC/"
            self.SAMP_Num_coeff = np.double(child.find("SAMP_NUM_COEFF").text.split())
            self.SAMP_Den_coeff = np.double(child.find("SAMP_DEN_COEFF").text.split())
            self.LINE_Num_coeff = np.double(child.find("LINE_NUM_COEFF").text.split())
            self.LINE_Den_coeff = np.double(child.find("LINE_DEN_COEFF").text.split())

        elif sensor =="Pleiades":
            path = "./Rational_Function_Model/Global_RFM/RFM_Validity/"
            for i in range(20):
                self.SAMP_Num_coeff.append(np.double(child[i].text))
                self.SAMP_Den_coeff.append(np.double(child[i + 20].text))
                self.LINE_Num_coeff.append(np.double(child[i + 40].text))
                self.LINE_Den_coeff.append(np.double(child[i + 60].text))
        else:
            print("Sensor not available")

        self.LONG_SCALE = np.double(child.find("LONG_SCALE").text)
        self.LONG_OFF = np.double(child.find("LONG_OFF").text)
        self.LAT_SCALE = np.double(child.find("LAT_SCALE").text)
        self.LAT_OFF = np.double(child.find("LAT_OFF").text)
        self.HEIGHT_SCALE = np.double(child.find("HEIGHT_SCALE").text)
        self.HEIGHT_OFF = np.double(child.find("HEIGHT_OFF").text)
        # Read I;J Scale and Offset
        self.SAMP_SCALE = np.double(child.find("SAMP_SCALE").text)
        self.SAMP_OFF = np.double(child.find("SAMP_OFF").text)
        self.LINE_SCALE = np.double(child.find("LINE_SCALE").text)
        self.LINE_OFF = np.double(child.find("LINE_OFF").text)

    def read_from_PRISMA_h5(self, h5_file, panchromatic=False):

        sensor = "PRISMA"

        f = h5py.File(h5_file, 'r')

        if panchromatic:
            rpc = f['HDFEOS']['SWATHS']['PRS_L2C_PCO']['Geocoding Model']
        else:
            rpc = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geocoding Model']

        self.SAMP_Num_coeff = rpc.attrs.get('SAMP_NUM_COEFF')
        self.SAMP_Den_coeff = rpc.attrs.get('SAMP_DEN_COEFF')
        self.LINE_Num_coeff = rpc.attrs.get('LINE_NUM_COEFF')
        self.LINE_Den_coeff = rpc.attrs.get('LINE_DEN_COEFF')

        self.LONG_SCALE = rpc.attrs.get('LONG_SCALE')
        self.LONG_OFF = rpc.attrs.get('LONG_OFF')
        self.LAT_SCALE = rpc.attrs.get('LAT_SCALE')
        self.LAT_OFF =  rpc.attrs.get('LAT_OFF')
        self.HEIGHT_SCALE = rpc.attrs.get('HEIGHT_SCALE')
        self.HEIGHT_OFF = rpc.attrs.get('HEIGHT_OFF')

        self.SAMP_SCALE = rpc.attrs.get('SAMP_SCALE')
        self.SAMP_OFF = rpc.attrs.get('SAMP_OFF')
        self.LINE_SCALE = rpc.attrs.get('LINE_SCALE')
        self.LINE_OFF = rpc.attrs.get('LINE_OFF')

    def update_RPC_ENVI(self, ENVI_input_file, ENVI_output_file):

        with io.open(ENVI_input_file, mode="r", encoding="latin-1") as fd:
            content = fd.read()
        with io.open(ENVI_output_file, mode="w", encoding="utf-8") as fd:
            fd.write(content)

        with fileinput.FileInput(ENVI_output_file, inplace=True) as file:
            n1 = 0; n2 = 0; n3 = 0; n4 = 0;
            for line in file:
                dd = line.split("=")
                if "Line Numerator" in dd[0]:
                    print("Line Numerator " + '{}'.format(n1+1) + " = " + '{:.12e}'.format(self.LINE_Num_coeff[n1]))
                    n1 = n1+1
                elif "Line Denominator" in dd[0]:
                    print("Line Denominator " + '{}'.format(n2+1) + " = " + '{:.12e}'.format(self.LINE_Den_coeff[n2]))
                    n2 = n2+1
                elif "Sample Numerator" in dd[0]:
                    print("Sample Numerator " + '{}'.format(n3+1) + " = " + '{:.12e}'.format(self.SAMP_Num_coeff[n3]))
                    n3 = n3+1
                elif "Sample Denominator" in dd[0]:
                    print("Sample Denominator " + '{}'.format(n4+1) + " = " + '{:.12e}'.format(self.SAMP_Den_coeff[n4]))
                    n4 = n4+1
                else:
                    print(line) #PYTHON # CONV, end="")

    def write_RPC(self, filename):
        with open(filename, mode="w") as fd:

            fd.writelines("LINE_OFF: \t\t" + '{:18.9f}'.format(self.LINE_OFF) + " pixels\n")
            fd.writelines("SAMP_OFF: \t\t" + '{:18.9f}'.format(self.SAMP_OFF) + " pixels\n")
            fd.writelines("LAT_OFF: \t\t" + '{:18.12e}'.format(self.LAT_OFF) + " degrees\n")
            fd.writelines("LONG_OFF: \t\t" + '{:18.12e}'.format(self.LONG_OFF) + " degrees\n")
            fd.writelines("HEIGHT_OFF: \t" + '{:18.6f}'.format(self.HEIGHT_OFF) + " meters\n")

            fd.writelines("LINE_SCALE: \t" + '{:18.9f}'.format(self.LINE_SCALE) + " pixels\n")
            fd.writelines("SAMP_SCALE: \t" + '{:18.9f}'.format(self.SAMP_SCALE) + " pixels\n")
            fd.writelines("LAT_SCALE: \t\t" + '{:18.12e}'.format(self.LAT_SCALE) + " degrees\n")
            fd.writelines("LONG_SCALE: \t" + '{:18.12e}'.format(self.LONG_SCALE) + " degrees\n")
            fd.writelines("HEIGHT_SCALE: \t" + '{:18.9f}'.format(self.HEIGHT_SCALE) + " metres\n")

            fd.writelines("LINE_NUM_COEFF_" + '{}'.format(i + 1) + ":\t" + '{:18.15e}'.format(coeff) + "\n" for i, coeff in
                          enumerate(self.LINE_Num_coeff))
            fd.writelines("LINE_DEN_COEFF_" + '{}'.format(i + 1) + ":\t" + '{:18.15e}'.format(coeff) + "\n" for i, coeff in
                          enumerate(self.LINE_Den_coeff))
            fd.writelines("SAMP_NUM_COEFF_" + '{}'.format(i + 1) + ":\t" + '{:18.15e}'.format(coeff) + "\n" for i, coeff in
                          enumerate(self.SAMP_Num_coeff))
            fd.writelines("SAMP_DEN_COEFF_" + '{}'.format(i + 1) + ":\t" + '{:18.15e}'.format(coeff) + "\n" for i, coeff in
                          enumerate(self.SAMP_Den_coeff))

    def to_geotiff_dict(self):
        """
        Return a dictionary storing the RPC coefficients as GeoTIFF tags.
        This dictionary d can be written in a GeoTIFF file header with:
            with rasterio.open("/path/to/image.tiff", "r+") as f:
                f.update_tags(ns="RPC", **d)
        """
        d = {}
        d["LINE_OFF"] = self.LINE_OFF
        d["SAMP_OFF"] = self.SAMP_OFF
        d["LAT_OFF"] = self.LAT_OFF
        d["LONG_OFF"] = self.LONG_OFF
        d["HEIGHT_OFF"] = self.HEIGHT_OFF

        d["LINE_SCALE"] = self.LINE_SCALE
        d["SAMP_SCALE"] = self.SAMP_SCALE
        d["LAT_SCALE"] = self.LAT_SCALE
        d["LONG_SCALE"] = self.LONG_SCALE
        d["HEIGHT_SCALE"] = self.HEIGHT_SCALE

        d["LINE_NUM_COEFF"] = " ".join([str(x) for x in self.LINE_Num_coeff])
        d["LINE_DEN_COEFF"] = " ".join([str(x) for x in self.LINE_Den_coeff])
        d["SAMP_NUM_COEFF"] = " ".join([str(x) for x in self.SAMP_Num_coeff])
        d["SAMP_DEN_COEFF"] = " ".join([str(x) for x in self.SAMP_Den_coeff])

        return {k: d[k] for k in sorted(d)}

    def compute_IJ(self, Lon, Lat, H):
        # Normalize the input coordinates
        L = (Lon - self.LONG_OFF) / self.LONG_SCALE
        P = (Lat - self.LAT_OFF) / self.LAT_SCALE
        H = (H - self.HEIGHT_OFF) / self.HEIGHT_SCALE

        MLT = self.compute_MLT(L, P, H)
        Rn = np.sum(MLT * np.array(self.LINE_Num_coeff)) / np.sum(MLT * np.array(self.LINE_Den_coeff))
        Cn = np.sum(MLT * np.array(self.SAMP_Num_coeff)) / np.sum(MLT * np.array(self.SAMP_Den_coeff))

        row = Rn * self.LINE_SCALE + self.LINE_OFF
        col = Cn * self.SAMP_SCALE + self.SAMP_OFF
        return (row, col)

    def normalize_coords(self, Lon, Lat, H):
        # Normalize the input coordinates
        L = (Lon - self.LONG_OFF) / self.LONG_SCALE
        P = (Lat - self.LAT_OFF) / self.LAT_SCALE
        H = (H - self.HEIGHT_OFF) / self.HEIGHT_SCALE
        return L, P, H

    def normalize_IJ(self, row, col):
        Rn = (row - self.LINE_OFF) / self.LINE_SCALE
        Cn = (col - self.SAMP_OFF) / self.SAMP_SCALE
        return Rn, Cn

    def compute_MLT(self, L, P, H):

        MLT = [1.0, L, P, H, L * P,
               L * H, P * H, L * L, P * P, H * H,
               P * L * H, L * L * L, L * P * P, L * H * H, L * L * P,
               P * P * P, P * H * H, L * L * H, P * P * H, H * H * H]

        return np.array(MLT)

    def GCP_assessment(self, data, verbose=True, geoid_flag=False):
        # data = pd.read_csv("GCPs.csv")
        Resid_Row = []
        Resid_Col = []

        geoid = 0
        # L, P, H = self.normalize_coords(data["Lon"][i], data["Lat"][i], data["H"][i])
        if geoid_flag:
            geoid = geo.geoidHeight2(data["Lat"].mean(), data["Lon"].mean())[0]
            if verbose: print("Geoid value used:", geoid)

        for i in np.arange(data.shape[0]):

            row_rpc, col_rpc = self.compute_IJ(data["Lon"][i], data["Lat"][i], data["H"][i]-geoid)

            Resid_Row.append(data["Row"][i] - row_rpc)
            Resid_Col.append(data["Col"][i] - col_rpc)

        Row_mean = np.mean(Resid_Row)
        Row_std = np.std(Resid_Row)

        Col_mean = np.mean(Resid_Col)
        Col_std = np.std(Resid_Col)

        if (verbose):
            print("--- Residual errors N_GCP:", data.shape[0], " ----")
            print("Row_mean:", Row_mean, "Row_std:", Row_std)
            print("Col_mean:", Col_mean, "Col_std:", Col_std)

        return Row_mean, Row_std, Col_mean, Col_std

    def GCP_refinement(self, data, verbose=True):

        A = np.zeros([2 * data.shape[0], 6])
        TN = np.zeros([2 * data.shape[0], 1])

        for i in np.arange(data.shape[0]):
            L, P, H = self.normalize_coords(data["Lon"][i], data["Lat"][i], data["H"][i])
            row_rpc, col_rpc = self.compute_IJ(data["Lon"][i], data["Lat"][i], data["H"][i])

            Rn_rpc, Cn_rpc = self.normalize_IJ(row_rpc, col_rpc)

            MLT_i = self.compute_MLT(L, P, H)
            LINE_Den_i = np.sum(MLT_i * np.array(self.LINE_Den_coeff))
            SAMP_Den_i = np.sum(MLT_i * np.array(self.SAMP_Den_coeff))

            A[2 * i, 0] = 1.0 / LINE_Den_i
            A[2 * i, 1] = L / LINE_Den_i
            A[2 * i, 2] = P / LINE_Den_i

            A[2 * i + 1, 3] = 1.0 / SAMP_Den_i
            A[2 * i + 1, 4] = L / SAMP_Den_i
            A[2 * i + 1, 5] = P / SAMP_Den_i

            Rn_gcp, Cn_gcp = self.normalize_IJ(data["Row"][i], data["Col"][i])

            TN[2 * i, 0] = Rn_gcp - Rn_rpc
            TN[2 * i + 1, 0] = Cn_gcp - Cn_rpc

        x, resid, rnk, singvals = la.lstsq(A, TN, rcond=None)

        # Update the rpc coeff
        self.LINE_Num_coeff[0] += x[0][0]
        self.LINE_Num_coeff[1] += x[1][0]
        self.LINE_Num_coeff[2] += x[2][0]

        self.SAMP_Num_coeff[0] += x[3][0]
        self.SAMP_Num_coeff[1] += x[4][0]
        self.SAMP_Num_coeff[2] += x[5][0]

    def SAR_RPC_Generator(self, SAR_Model, savegrid = "SAR_reference_Grid.csv"):
        print("test rpc generator")

        Grid_Lat_Dim = 10
        Grid_Lon_Dim = 10
        Grid_H_Dim = 5

        minLon = np.min([SAR_Model.corner1["Lon"],SAR_Model.corner2["Lon"],SAR_Model.corner3["Lon"],SAR_Model.corner4["Lon"]])
        maxLon = np.max([SAR_Model.corner1["Lon"], SAR_Model.corner2["Lon"], SAR_Model.corner3["Lon"], SAR_Model.corner4["Lon"]])

        minLat = np.min([SAR_Model.corner1["Lat"], SAR_Model.corner2["Lat"], SAR_Model.corner3["Lat"], SAR_Model.corner4["Lat"]])
        maxLat = np.max([SAR_Model.corner1["Lat"], SAR_Model.corner2["Lat"], SAR_Model.corner3["Lat"], SAR_Model.corner4["Lat"]])

        print(minLon, maxLon)
        print(minLat, maxLat)

        Grid_Lat_Steps = minLat + np.arange(Grid_Lat_Dim+1)*((maxLat-minLat)/Grid_Lat_Dim)
        Grid_Lon_Steps = minLon + np.arange(Grid_Lon_Dim+1)*((maxLon-minLon)/Grid_Lon_Dim)
        Grid_H_Steps = -Grid_H_Dim*50+SAR_Model.avg_height + np.arange(2*Grid_H_Dim)*50

        #print(Grid_Lat_Steps)
        #print(Grid_Lon_Steps)
        #print(Grid_H_Steps)

        Grid = []

        for lat in Grid_Lat_Steps:
            for lon in Grid_Lon_Steps:
                for h in Grid_H_Steps:

                    i, j, mod = SAR_Model.GroundToSlant(lat,lon,h)

                    buffer = 1000

                    if i > +buffer and i < SAR_Model.width-buffer:
                        if j > +buffer and j < SAR_Model.height-buffer:
                            Grid.append({"i": i, "j": j, "Lat": lat, "Lon": lon, "H":h })

        data_grid = pd.DataFrame(Grid)

        data_grid_validation = pd.DataFrame(Grid)

        data_grid.rename(columns={'i' : 'Col', 'j' : 'Row' }).to_csv(savegrid)

        data_grid_validation.plot(kind='scatter', x='Lat', y='Lon', color='red')
        plt.savefig('lat_lon_grid.png')

        data_grid_validation.plot(kind='scatter', x='i', y='j', color='red')
        plt.savefig('i_j_grid.png')


        print(data_grid.head())

        #print(data_grid["Lat"].min(), data_grid["Lat"].max())
        #print(data_grid["Lon"].min(), data_grid["Lon"].max())

        self.LONG_OFF = data_grid["Lon"].min()
        self.LONG_SCALE = data_grid["Lon"].max()-data_grid["Lon"].min()

        self.LAT_OFF = data_grid["Lat"].min()
        self.LAT_SCALE = data_grid["Lat"].max()-data_grid["Lat"].min()

        self.HEIGHT_OFF = data_grid["H"].min()
        self.HEIGHT_SCALE = data_grid["H"].max()-data_grid["H"].min()

        self.SAMP_OFF = 1.0
        self.SAMP_SCALE = data_grid["i"].max()-data_grid["i"].min()-1

        self.LINE_OFF = 1.0
        self.LINE_SCALE = data_grid["j"].max()-data_grid["j"].min()-1

        data_grid["Lon"] = (data_grid["Lon"]-self.LONG_OFF)/self.LONG_SCALE
        data_grid["Lat"] = (data_grid["Lat"]-self.LAT_OFF) / self.LAT_SCALE
        data_grid["H"] = (data_grid["H"]-self.HEIGHT_OFF) / self.HEIGHT_SCALE
        data_grid["i"] = (data_grid["i"]-self.SAMP_OFF) / self.SAMP_SCALE
        data_grid["j"] = (data_grid["j"]-self.LINE_OFF)/ self.LINE_SCALE

        print(data_grid.info())
        print(data_grid.head())

        data_grid.plot(kind='scatter', x='Lat', y='Lon', color='red')
        #plt.show()
        plt.savefig('normalize_grid.png')

        def inputLS(grid, tagij):

            N = grid.shape[0]

            print("Number of points: ", N)

            A = np.zeros((N,39))

            TN = np.zeros((N,1))

            #idx = 0

            for idx, point in grid.iterrows():

                TN[idx][0] = point[tagij]

                Poly_Der = np.array([1,
                            point["Lon"],
                            point["Lat"],
                            point["H"],
                            point["Lon"] * point["Lat"],
                            point["Lon"] * point["H"],
                            point["Lat"] * point["H"],
                            point["Lon"] * point["Lon"],
                            point["Lat"] * point["Lat"],
                            point["H"] * point["H"],
                            point["Lon"] * point["Lat"] * point["H"],
                            point["Lon"] * point["Lon"] * point["Lon"],
                            point["Lon"] * point["Lat"] * point["Lat"],
                            point["Lon"] * point["H"] * point["H"],
                            point["Lat"] * point["Lon"] * point["Lon"],
                            point["Lat"] * point["Lat"] * point["Lat"],
                            point["Lat"] * point["H"] * point["H"],
                            point["H"] * point["Lon"] * point["Lon"],
                            point["H"] * point["Lat"] * point["Lat"],
                            point["H"] * point["H"] * point["H"]])

                #concatente num and den derivates
                A[idx] = np.concatenate((-1.0*Poly_Der, point[tagij]*Poly_Der[1:]))

            print("first line A")
            print(A[1])

            print("first line TN")
            print(TN[1])

            return TN, A

        # def svd_solve(a, b):
        #     [U, s, Vt] = la.svd(a, full_matrices=False)
        #     r = max(np.where(s >= 1e-12)[0])
        #     temp = np.dot(U[:, :r].T, b) / s[:r]
        #     return np.dot(Vt[:r, :].T, temp)
        #
        # x = svd_solve(A1,TN1)

        TN1, A1 = inputLS(data_grid, "i")
        model = sm.GLS(-1*TN1, A1)

        results = model.fit() #method="qr")
        #print(results.summary())
        #print(results.params)

        self.SAMP_Num_coeff = results.params[0:20]
        self.SAMP_Den_coeff = np.concatenate(([1.0], results.params[20:]))

        print("SAMP parameters", type(self.SAMP_Num_coeff))
        print(self.SAMP_Num_coeff)
        print(self.SAMP_Den_coeff)

        TN2, A2 = inputLS(data_grid, "j")
        model2 = sm.GLS(-1*TN2, A2)

        results2 = model2.fit() #method="qr")
        # print(results.summary())
        # print(results.params)

        self.LINE_Num_coeff = results2.params[0:20]
        self.LINE_Den_coeff = np.concatenate(([1.0], results2.params[20:]))

        print("LINE parameters", type(self.LINE_Num_coeff))
        print(self.LINE_Num_coeff)
        print(self.LINE_Den_coeff)

        #ASSESSMENT RPC

        data_grid_validation = data_grid_validation.rename(columns={"i": "Col", "j": "Row"})

        print(data_grid_validation.head())

        self.GCP_assessment(data_grid_validation,True)

        #model2 = sm.GLS(TN1, A1)
        #results2 = model2.fit()
        # if (verbose):
        #print(results2.summary())

        # def svd_solve(a, b):
        #     [U, s, Vt] = la.svd(a, full_matrices=False)
        #     r = max(np.where(s >= 1e-12)[0])
        #     temp = np.dot(U[:, :r].T, b) / s[:r]
        #     return np.dot(Vt[:r, :].T, temp)
        #
        # x = svd_solve(A1,TN1)

        #print(results2.params)

        #self.SAMP_Num_coeff
        #self.LINE_Num_coeff

        #x2, resid, rnk, singvals = la.lstsq(A1, TN1, rcond=None)

        #print(x2)
