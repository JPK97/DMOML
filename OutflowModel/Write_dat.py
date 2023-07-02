## Import used functions
import pandas as pd
import numpy as np
import platform
from datetime import datetime as datetime


def WriteData(nc=10, path="./", data_file="Random_Database_%s.dat" %(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))):

    ## Database header
    header = ["Model name", "Theta [deg]", "Phi [deg]", "v [km/s]", "n [cm-3]", "T [K]", "alpha", "dmin [au]", "sigmaof", "ThetaMax [deg]", "epsilon", "gamma", "M1", "beta", "r0 [au]", "z0 [au]", "Label"]

    ## Name of the machine where the files will be generated
    label = platform.node()

    ## To be filled database
    data = []

    ## Value ranges for Cabrit model
    Thetae = [0, 90]
    Phie = [0, 360]
    ve = [5,20]
    ne = [1e-1,5e-1]
    Te = [25,100]
    alphae = [0,-2]
    dmine = [15000, 25000]
    sigmae = [4,6]
    ThetaMaxe = [10,40]

    ## Generate nc random values in the ranges
    Thetas = np.random.uniform(Thetae[0], Thetae[1], nc)
    Phis = np.random.uniform(Phie[0], Phie[1], nc)
    vs = np.random.uniform(ve[0], ve[1], nc)
    ns = np.random.uniform(ne[0], ne[1], nc)
    Ts = np.random.uniform(Te[0], Te[1], nc)
    alphas = np.random.uniform(alphae[0], alphae[1], nc)
    dmins = np.random.uniform(dmine[0], dmine[1], nc)
    sigmas = np.random.uniform(sigmae[0], sigmae[1], nc)
    ThetaMaxs = np.random.uniform(ThetaMaxe[0], ThetaMaxe[1], nc)
    epsilons = np.zeros(nc) + 1
    gammas = np.zeros(nc) + 1.4
    M1s = np.zeros(nc) + 20
    betas = np.zeros(nc) - 4
    r0s = np.zeros(nc) + 10000
    z0s = np.zeros(nc) + 10000000
    labels = [label] * nc

    ## Initiallize the database
    df = pd.DataFrame(index=["Sample %i" %(i) for i in range(len(data))], columns=header)

    ## Fill the database
    df["Model name"] = ["Cabrit"]*nc
    df["Theta [deg]"] = Thetas
    df["Phi [deg]"] = Phis
    df["v [km/s]"] = vs
    df["n [cm-3]"] = ns
    df["T [K]"] = Ts
    df["alpha"] = alphas
    df["dmin [au]"] = dmins
    df["sigmaof"] = sigmas
    df["ThetaMax [deg]"] = ThetaMaxs
    df["epsilon"] = epsilons
    df["gamma"] = gammas
    df["M1"] = M1s
    df["beta"] = betas
    df["r0 [au]"] = r0s
    df["z0 [au]"] = z0s
    df["Label"] = labels

    ## Save the database
    df.to_csv(path+data_file)

    ## Return the directory and name of the database
    return path+data_file
