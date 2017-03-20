import sys
import os


from DBLR_lab import BLR_ACC3_i
from DBLR_lab import energy_measure
from DBLR_lab import find_baseline
import multiprocessing as mp
import numpy as np
from time import time
from functools import partial
from cal_IO_lib import get_CALHF




def BLR_lambda(signal_daq,**kwarg):

    # BLR Wrapper

    coef = kwarg.get('coef')                # BLR Coeff
    thr = kwarg.get('thr')                  # thr Accum flush threshold
    acum_FLOOR = kwarg.get('acum_FLOOR')    # Low limit for accumulator
    coef_clean = kwarg.get('coef_clean')    # Filter Coeff
    filter = kwarg.get('filter',"True")     # Filter applied
    i_factor = kwarg.get('i_factor',1)      # Interpolation factor
    e_thr = kwarg.get('e_thr',5)            # Threshold for energy computing
    SPE = kwarg.get('SPE',20)               # pe integrated value


    energy = np.array([])

    recons,a,b=BLR_ACC3_i(signal_daq,
                          coef,thr,acum_FLOOR,coef_clean,filter, i_factor)

    LIMIT_L_n,LIMIT_H_n,energy = energy_measure(recons,
                                                threshold=e_thr,
                                                SPE=SPE)

    output={'recons':recons,            # Reconstructed signal
            'acum':a,                   # Internal DBLR accumulator
            'signal_daq':b,             # input signal after offset comp.
            'LIMIT_L':LIMIT_L_n,        # Pos. Edge crossing point for energy thr
            'LIMIT_H':LIMIT_H_n,        # Neg. Edge crossing point for energy thr
            'ENERGY':energy             # Energy Measurement
            }

    return output



def BLR_batch(db_path,**kwarg):

    # BLR batch wrapper
    PMT={}
    PMT['coef']        = kwarg.get('coef')        # BLR Coeff
    PMT['thr']         = kwarg.get('thr')         # thr Accum flush threshold
    PMT['acum_FLOOR']  = kwarg.get('acum_FLOOR')  # Low limit for accumulator
    PMT['coef_clean']  = kwarg.get('coef_clean')  # Filter Coeff
    PMT['filter']      = kwarg.get('filter')      # Filter applied
    PMT['i_factor']    = kwarg.get('i_factor')    # Interpolation factor
    PMT['e_thr']       = kwarg.get('e_thr')       # Threshold for energy computing
    PMT['SPE']         = kwarg.get('SPE')         # pe integrated value

    pmt_number         = kwarg.get('point')       # PMT channel number
    n_events           = kwarg.get('n_events')    # Number of events in batch


    dbase = get_CALHF(db_path)
    f = dbase[pmt_number,:,:]

    # Multiprocess Work
    pool_size = mp.cpu_count()
    pool = mp.Pool(processes=pool_size)

    mapfunc = partial(BLR_lambda, **PMT)

    pool_output = pool.map(mapfunc, (f[:,i] for i in range(n_events)))

    pool.close()
    pool.join()

    ENERGY=np.zeros(n_events)
    for j in range(n_events): ENERGY[j]=pool_output[j]['ENERGY']


    return ENERGY


#############################################################################

def main():

    PATH_F = "/mnt/WINDOWS_ntfs/"

    n_events = 500
    PMT = 2
    dbase = get_CALHF(PATH_F+'/DATOS_DAC/CALIBRATION/cal_50u.h5.z')
    f = np.array(dbase[PMT,:,:])
    PMT_info = {'coef':1.65E-3,
                'thr':1.0,
                'acum_FLOOR':1.0,
                'coef_clean':1E-6,
                'filter':True,
                'i_factor':1.0,
                'e_thr':5.0,
                'SPE':20.0}



    tic = time()
    pool_size = mp.cpu_count()/2

    pool = mp.Pool(processes=pool_size)

    mapfunc = partial(BLR_lambda, **PMT_info)

    mapfunc(f[:,2])

    pool_output = pool.map(mapfunc, (f[:,i] for i in range(n_events)))

    pool.close()
    pool.join()

    ENERGY=np.zeros(n_events)
    for j in range(n_events): ENERGY[j]=pool_output[j]['ENERGY']

    tac = time()-tic

    print ENERGY

    print "MP time =",tac

    tic = time()
    for i in range(n_events):
        BLR_lambda(f[:,i],**PMT_info)
    tac = time()-tic
    print "No MP time =",tac

if __name__=="__main__":
    main()
