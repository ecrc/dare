#  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                      All rights reserved.

from asyncore import write
import numpy as np
from os.path import join
from astropy.io.fits import open as fitsopen
from astropy.io.fits import writeto as fitswrite
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, help='data path')

def openfits(filename):
    return (fitsopen(filename)[0].data)

def writefits(filename, fitsfile):
    fitswrite(filename,fitsfile,overwrite=True)

if __name__ == "__main__":
    args = parser.parse_args()
    print("args", args.datapath)
    Binitfile = join(args.datapath, "Btinitial.fits")
    Rtfile = join(args.datapath, "Rt.fits")
    Btinitial = openfits(Binitfile)
    Rt = openfits(Rtfile)
    if Btinitial.shape[0] != Rt.shape[0]:
        print("dim not equals!")
    for i in range(Rt.shape[0]):
        Btinitial[i,:] /= np.sqrt(Rt[i,i])
    # write BinvR.fits
    fitswrite(join(args.datapath,"BinvRt.fits"),Btinitial.T,overwrite=True)

    
