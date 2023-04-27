from rpcmodel import RPCmodel


def compute_and_assess(Path_RPC, Path_GCP, output_path, verbose=False):

    if verbose:
        print("Input RPC File: " + Path_RPC)
        print("Input GCP File: " + Path_GCP + "\n")

    rpc_img1 = RPCmodel()
    rpc_img1.read_from_ENVI(Path_RPC)

    data = pd.read_csv(Path_GCP)
    data = data.dropna(axis=1)
    N_GCP = data.shape[0]

    if N_GCP < 3:
        print("Required at least 3 GCPs --> procedure aborted")
        exit(-1)

    if verbose:
        print("Number of available GCPs:", N_GCP)
        print(data.head())
        # Assess the RPC before the refinement with all the available GCPs
        print("")
        print("Original RPC Model assessment")

    Ori_Row_mean, Ori_Row_std, Ori_Col_mean, Ori_Col_std = rpc_img1.GCP_assessment(data, verbose)

    rpc_img1.GCP_refinement(data, verbose)

    if verbose:
        print("")
        print("--- Refined RPC Model assessment ---")

    Ref_Row_mean, Ref_Row_std, Ref_Col_mean, Ref_Col_std = rpc_img1.GCP_assessment(data, verbose)

    if verbose:
        print("")
        print("--- Leave one out (K-fold) validation ---")

        kf = KFold(n_splits=N_GCP)
        kf.get_n_splits(data)
        results_df = pd.DataFrame()

        for train_index, test_index in kf.split(data):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            rpc_img_n = rpc_model()
            rpc_img_n.read_from_ENVI(Path_RPC)
            rpc_img_n.GCP_refinement(X_train.reset_index(), verbose=False)
            Row_mean, Row_std, Col_mean, Col_std = rpc_img_n.GCP_assessment(X_test.reset_index(), verbose=False)
            X_test_copy = X_test.copy()
            X_test_copy['D_Col'] = [Col_mean]
            X_test_copy['D_Row'] = [Row_mean]
            X_test_copy['Mod'] = [np.sqrt(Row_mean * Row_mean + Col_mean * Col_mean)]
            results_df = results_df.append(X_test_copy)

        print(results_df.head(15))
        print("")
        print("--- Leave One Out Residual errors N_GCP:", N_GCP, " ----")
        print("Row_mean:", results_df[['D_Row']].mean(axis=0).to_numpy()[0], "Row_std:",
              results_df[['D_Row']].std(axis=0).to_numpy()[0])
        print("Col_mean:", results_df[['D_Col']].mean(axis=0).to_numpy()[0], "Col_std:",
              results_df[['D_Col']].std(axis=0).to_numpy()[0])

    print("")
    print("Generated Refined RPC file:" + output_path)
    rpc_img1.update_RPC_ENVI(Path_RPC, output_path)


if __name__ == '__main__':

    default_path = os.getcwd()

    parser = argparse.ArgumentParser(
        description='This software refines the RPC coefficient using a set of at least 3 GCPs')
    parser.add_argument('RPC_Path', help='Insert the path to the RPC file in ENVI metadata format')
    parser.add_argument('GCP_Path',
                        help='Insert the path to the GCP file in CSV format with \n' +
                             'the required columns header: Lat, Lon, H, Col, Row, id ')
    parser.add_argument("-o","--output", help="insert the output file path (default: current_dir/refined_RPC.txt)", default=default_path+"/refined_RPC.txt")
    parser.add_argument("-v", "--verbosity", help="increase output verbosity", action="store_true")

    args = parser.parse_args()

    compute_and_assess(args.RPC_Path, args.GCP_Path, args.output, args.verbosity)