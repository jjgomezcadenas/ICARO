# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 15:18:42 2016

@author: viherbos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tables as tb
import multiprocessing as mp
from functools import partial
import os



def read_CALHF(in_path, PMT, event):
	#Reads PMT signal realted to a given event from HDF5 file
	a = pd.read_hdf(in_path,'data')
	b = np.array(a[event,PMT,:])
	# WATCHOUT no transpose done
	return b

def read_DATE_hdf5(in_path, PMT, event):
	#Reads PMT signal realted to a given event from HDF5 file
    a = pd.HDFStore(in_path)
    b = np.array(a.root.RD.pmtrwf)
    c = b[event,PMT,:]
    return c

def get_CALHF(in_path):
	#Reads PMT signal related to a given event from HDF5 file
    b = np.array(pd.read_hdf(in_path,'data'))
    c = b.transpose(1,2,0)
    return c

def get_DATE_hdf5(in_path):
	#Reads PMT signal related to a given event from HDF5 file
    a = pd.HDFStore(in_path)
    b = np.array(a.root.RD.pmtrwf)
    c = b.transpose(1,2,0)
    return c



def DATE_to_CALHF(**kwarg):

	# Reads signals from each event and PMT stored in HF5 DATE
	# Creates HDF5 based on a single PANEL node where Item=PMT /
	# minor_axis(column)=Event / major_axis=sample
	# Output file holds all the channel outputs for a given pulse length

	box				= kwarg.get('box_number',0)					# FEE_BOX number
	output_path		= kwarg.get('output_path','F:/DATOS_DAC/CALIBRATION_FEB_2017/')   			# Ouptut path
	input_path		= kwarg.get('input_path','F:/DATOS_DAC/CALIBRATION_FEB_2017/raw_IFIC_data/')   # Input path
	pulse_length	= kwarg.get('pulse_length',[1,2,4,6,8,10,
												12,14,16,18,
												20,25,30,35,
												40,45,50,60,
												70,80,90,100])	# Pulse Lenght (useconds)


	files=[]
	#Stores all the files
	files_hdf5=[]
	#Stores hdf5 files

	for (dirpath, dirnames, filenames) in os.walk(input_path):
	    files.extend(filenames)
	    break

	for i in files:
	    if (('.h5' in i) and (('box'+str(box)) in i)):
	        files_hdf5.append(i)
	# All the hdf5 files in the given directory

	files_box={}

	for j in pulse_length:
	    files_box[str(j)]=[]

	    for i in files_hdf5:
	        # Dictionary to store pulse_length files using length index
	        if (('_'+str(j)+'u') in i) and ('_ch' in i):
	            files_box[str(j)].append(i)
	print input_path
	print files
	print files_box
	######################################################################




	#The following reads all the data and stores it in a Panel
	for j in pulse_length:
		# Read all the files by pulse length

		out_file = ('box' + str(box) + '_' + str(j) + 'u.h5')

		if os.path.exists((output_path+out_file)):
			os.remove((output_path+out_file))

		store = pd.HDFStore((output_path+out_file), complevel=9, complib='zlib')

		data_aux=tb.open_file(input_path + files_box[str(j)][0],"r")
		raw_data_aux = np.copy(data_aux.root.RD.pmtrwf)
		raw_data_aux = 0*raw_data_aux
		data = pd.Panel(raw_data_aux)
		# Creates empty Panel with same structure as DATE HF5
		# Panel structure -> Items = Events //
		#	  				 Major_axis = channel //
		#					 Minor_axis = samples

		for i in files_box[str(j)]:

			# Beginning and End of channel number
			b_ch = i.find('_ch')+len('_ch')
			e_ch = i[b_ch:].find('_')+b_ch
			# Channel Number
			ch_number = int(i[b_ch:e_ch])
			print i,'    channel',ch_number,'    pulse_length',j

			data_aux=tb.open_file(input_path + i,"r")
			raw_data = data_aux.root.RD.pmtrwf
			raw_data_aux[:,ch_number,:]=np.array(raw_data[:,0,:])
        	# Put data in its place

		store.put('data',data)
		# Dumps data to file
		store.close()




###################### MULTIPROCESS VERSION ####################################

def CALHF_lambda(j,**kwarg):
	# j Array of pulse lengths
	box 		= kwarg.get('box')			# FEE Box
	output_path = kwarg.get('output_path')
	input_path  = kwarg.get('input_path')
	files_box   = kwarg.get('files_box')

	# Read all the files by pulse length
	out_file = ('box' + str(box) + '_' + str(j) + 'u.h5')

	if os.path.exists((output_path+out_file)):
		os.remove((output_path+out_file))

	store = pd.HDFStore((output_path+out_file), complevel=5, complib='zlib')

	data_aux=tb.open_file(input_path + files_box[str(j)][0],"r")
	raw_data_aux = np.copy(data_aux.root.RD.pmtrwf)
	raw_data_aux = 0*raw_data_aux
	data = pd.Panel(raw_data_aux)
	# Creates empty Panel with same structure as DATE HF5
	# Panel structure -> Items = Events //
	#	  				 Major_axis = channel //
	#					 Minor_axis = samples

	for i in files_box[str(j)]:

		# Beginning and End of channel number
		b_ch = i.find('_ch')+len('_ch')
		e_ch = i[b_ch:].find('_')+b_ch
		# Channel Number
		ch_number = int(i[b_ch:e_ch])
		print i,'    channel',ch_number,'    pulse_length',j

		data_aux=tb.open_file(input_path + i,"r")
		raw_data = data_aux.root.RD.pmtrwf
		raw_data_aux[:,ch_number,:]=np.array(raw_data[:,0,:])
    	# Put data in its place

	store.put('data',data)
	# Dumps data to file
	store.close()



def DATE_to_CALHF_MP(**kwarg):
	# Reads signals from each event and PMT stored in HF5 DATE
	# Creates HDF5 based on a single PANEL node where Item=PMT /
	# minor_axis(column)=Event / major_axis=sample
	# Output file holds all the channel outputs for a given pulse length

    box			= kwarg.get('box_number',0)					# FEE_BOX number
    output_path	= kwarg.get('output_path','F:/DATOS_DAC/CALIBRATION_FEB_2017/')   			# Ouptut path
    input_path	= kwarg.get('input_path','F:/DATOS_DAC/CALIBRATION_FEB_2017/raw_IFIC_data/')   # Input path
    pulse_length	= kwarg.get('pulse_length',[1,2,4,6,8,10,12,14,16,18,20,25,30,35,	40,45,50,60,70,80,90,100])
    MP = kwarg.get('MP',4)  # Number of processes


    files=[]
	#Stores all the files
    files_hdf5=[]
	#Stores hdf5 files

    for (dirpath, dirnames, filenames) in os.walk(input_path):
        files.extend(filenames)
        break

    for i in files:
        if (('.h5' in i) and (('box'+str(box)) in i)):
            files_hdf5.append(i)
	# All the hdf5 files in the given directory

    files_box={}

    for j in pulse_length:
        files_box[str(j)]=[]

        for i in files_hdf5:
	        # Dictionary to store pulse_length files using length index
            if (('_'+str(j)+'u') in i) and ('_ch' in i):
	            files_box[str(j)].append(i)

	print str(box)
    print input_path
    print files
    print files_box

    # Multiprocess Work
    pool_size = MP #mp.cpu_count()
    pool = mp.Pool(processes = pool_size)
    datum={}
    datum['box']= box

    datum['output_path'] = output_path
    datum['input_path']  = input_path
    datum['files_box']   = files_box
    mapfunc = partial(CALHF_lambda, **datum)

    pool_output = pool.map(mapfunc, pulse_length)
    pool.close()
    pool.join()





def main():

	DATE_to_CALHF_MP()

	# a = read_CALHF('/mnt/WINDOWS_ntfs/DATOS_DAC/CALIBRATION_FEB_2017/box0_100u.h5',3,10)
	# plt.plot(a)
	# plt.show()


if __name__ == "__main__":
	main()
