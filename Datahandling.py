from astropy.io import fits
import pandas as pd
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import glob, sys, os
import h5py
import time
import params


"""
- Read in .csv data files fo Gaia DR2 and convert relevant columns into .fits files.
  Run in 5 batches of each last digit of .csv file name (even numbers). For each batch
  loop over the second to last digit in file name. This does not use too much memory.
  Get 25 files per parameter. Need to merge the parameter files. Note the order the files 
  are read in.
""" 


def get_data(path, savepath, colnames):
    """
    Read data from .csv file and write it to .h5 files. Sending in parts of total data files.
    Input: - path, string,          path to the files to read
           - savepath, string,      path to the parameterfiles to save
           - colname, list, string  list with the column names to use in the .csv files

    """

    print('Read path:', path)
    print('Save path:', savepath)
    print('Use the columns:', colnames)


    if path.endswith('.csv.gz'):
        print('Read .csv files')
        datafiles = glob.glob(path)
        datafiles.sort()

        # Find the column numbers to use.
        if len(datafiles) > 0:
            colnum = getCols(datafiles[0], colnames)
            print('Use culumns numbers:', colnum)
        else:
            print('No .csv files with the current last digit')
            return None

    else:
        print('There are no .csv files in the path folder.')
        sys.exit()

    ###
    if len(datafiles) > 0:
        ncol = len(colnum)
        data = np.array([[] for i in range(len(colnum))])
        data = np.reshape(data, (0,ncol))

        time0 = time.time()
        for ind, file in enumerate(datafiles):
            
            dt = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=colnum)

            if ind > 0:
                data = np.append(data, dt, axis=0)

            else:
                data = dt

            ##    
            if ind%250 == 0:
                end = time.time()
                print('File number {} are read, used {} sec so far'.format(ind, (end-time0)))
            

        time1 = time.time()
        print('Time used:', time1-time0)
        print('Write data to files:')
        # Write to files:
        
        for i in range(ncol):
            print('Write file for {}'.format(colname[i]))
            if colname[i]=='ra' or colname[i]=='dec':
                # convert from degrees to radians
                data[:,i] = data[:,i]*params.DEG2RAD

            hdf = h5py.File('{}{}'.format(savepath, colname[i]), 'w')
            if isinstance(data[0,i], int) == True:
                print('{} is int'.format(colname[i]))
                hdf.create_dataset('{}'.format(colname), data=data[:,i].astype(int))
            elif isinstance(data[0,i], float) == True:
                print('{} is float'.format(colname[i]))
                hdf.create_dataset('{}'.format(colname), data=data[:,i].astype(dtype=np.float32))
            else:
                print('{} is string'.format(colname[i]))
                hdf.create_dataset('{}'.format(colname), data=data[:,i].astype(str))

            hdf.close()
        
        ############

    else:
        print('No files to read')
        return None
    # end get_data


def get_csv_data(read_path, save_path):
    """
    Read data from .csv file and write it to .h5 files. Sending in parts of total data files.
    
    """
    path = read_path 
    print('read path:', path)
    a = path.split('.')
    print('Save path:', save_path)
    
    if a[-2] == 'csv':
         print('Read .csv files')
         datafiles = glob.glob(path)
         datafiles.sort()
         print('Number of data files: {}'.format(len(datafiles)))

    else:
        print('There are no .csv files in the path folder.')
        sys.exit()
    

    #####
    if len(datafiles) > 0:
                
        cols = ['source_id', 'random_index', 'ra', 'ra_error','dec', 'dec_error', 'parallax',\
                'parallax_error', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']
 
        source_id    = np.array([])           
        random_index = np.array([]) 
        ra           = np.array([])
        ra_err       = np.array([])           
        dec          = np.array([])
        dec_err      = np.array([])
        parallax     = np.array([])
        parallax_err = np.array([])
        mean_mag_G   = np.array([])
        mean_mag_BP  = np.array([])
        mean_mag_RP  = np.array([])
               

        # read all files
        start = time.time()
        for ind, file in enumerate(datafiles):
            df = pd.read_csv(file, usecols=cols)
            data_array = np.asarray(df)
                       
            source_id    = np.append(source_id, data_array[:,0])
            random_index = np.append(random_index, data_array[:,1])
            ra           = np.append(ra, data_array[:,2])
            ra_err       = np.append(ra_err, data_array[:,3])
            dec          = np.append(dec, data_array[:,4])
            dec_err      = np.append(dec_err, data_array[:,5])
            parallax     = np.append(parallax, data_array[:,6])
            parallax_err = np.append(parallax_err, data_array[:,7])
            mean_mag_G   = np.append(mean_mag_G, data_array[:,8])
            mean_mag_BP  = np.append(mean_mag_BP, data_array[:,9])
            mean_mag_RP  = np.append(mean_mag_RP, data_array[:,10])
        
            if ind%250 == 0:
                end = time.time()
                print('File number {} are read, used {} sec so far'.format(ind, (end-start)))


        # convert angular position to radians
    
        ra  = ra*params.DEG2RAD
        dec = dec*params.DEG2RAD 
        
        # write parameter files
        
        hdf0 = h5py.File('{}_source_id.h5'.format(save_path), 'w')
        hdf0.create_dataset('source_id', data=source_id.astype(int))  
        hdf0.close()
        hdf1 = h5py.File('{}_random_index.h5'.format(save_path), 'w')
        hdf1.create_dataset('random index', data=random_index.astype(int))
        hdf1.close()
        hdf2 = h5py.File('{}_ra.h5'.format(save_path), 'w')
        hdf2.create_dataset('ra', data=ra.astype(dtype=np.float32))
        hdf2.close()
        hdf3 = h5py.File('{}_ra_error.h5'.format(save_path), 'w')
        hdf3.create_dataset('ra_error', data=ra_err.astype(dtype=np.float32))
        hdf3.close()
        hdf4 = h5py.File('{}_dec.h5'.format(save_path), 'w')
        hdf4.create_dataset('dec', data=dec.astype(dtype=np.float32))
        hdf4.close()
        hdf5 = h5py.File('{}_dec_error.h5'.format(save_path), 'w')
        hdf5.create_dataset('dec_error', data=dec_err.astype(dtype=np.float32))
        hdf5.close()
        hdf6 = h5py.File('{}_parallax.h5'.format(save_path), 'w')
        hdf6.create_dataset('parallax', data=parallax.astype(dtype=np.float32))
        hdf6.close()
        hdf7 = h5py.File('{}_parallax_error.h5'.format(save_path), 'w')
        hdf7.create_dataset('parallax error', data=parallax_err.astype(dtype=np.float32))
        hdf7.close()
        hdf8 = h5py.File('{}_mean_mag_G.h5'.format(save_path), 'w')
        hdf8.create_dataset('mean_mag_G', data=mean_mag_G.astype(dtype=np.float32))
        hdf8.close()
        hdf9 = h5py.File('{}_mean_mag_BP.h5'.format(save_path), 'w')
        hdf9.create_dataset('mean_mag_BP', data=mean_mag_BP.astype(dtype=np.float32))
        hdf9.close()
        hdf10 = h5py.File('{}_mean_mag_RP.h5'.format(save_path), 'w')
        hdf10.create_dataset('mean_mag_RP', data=mean_mag_RP.astype(dtype=np.float32))
        hdf10.close()

    else:
        print('No files to read')
        return None
    # end read_csv           


def fileMerger(in_files, name, out_file_name, type):
    """
    Merge the different datafiles for one parameter to one file, and then delete the old ones?
    """
    print('Merge parameter files for parameter: {}'.format(in_files))
    
    files = glob.glob('*{}.h5'.format(in_files))
    files.sort()
    
    Data  = np.array([])
    print('________')
                
    for file in files:
        print('Read file: {}'.format(file))
        read_file = h5py.File('{}'.format(file), 'r')
        dt        = np.asarray(read_file['{}'.format(name)])
        read_file.close()

        Data      = np.append(Data, dt)
        print('New data has length {}, total array lenght={}'.format(len(dt), len(data)))

        # delete input files
        if os.path.isfile(file):
            print('Delete file: {}'.format(file))
            os.remove(file)
        else:
            print('Error: {} is not found'.format(file))
    

    # Write new file:
    outdata = h5py.File('{}.h5'.format(out_file_name), 'w')

    if type == 'int':
        outdata.create_dataset('{}'.format(name), data=Data.astype(int))

    elif type=='float':
        outdata.create_dataset('{}'.format(name), data=Data.astype(dtype=np.float32))

    elif type=='str':
        outdata.create_dataset('{}'.format(name), data=Data.astype(str))

    outdata.close()
    # End fileMerger


def getCols(file, colname):
    """
    Get the culumn number of the csv.files. Need to know what column names you want!
    Input:  - file, string, a file form the .csv file list.
            - colname, list, the names of the columns to look at.
    """

    a = np.genfromtxt(file, delimiter=',', names=True)
    b = np.asarray(a.dtype.names)
    colnum = []
    for j in range(len(colname)):
        for i in range(len(b)):
            if colname[j] == b[i]:
                colnum.append(i)

    return colnum


def call_get_data(last_digit, colnames):
    """
    Call the function get_data() or det_csv_data() with the required settings.
    """

    directory = '/mn/stornext/d16/cmbco/eirikgje/data/gaia/cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/'
    savepath  = '/mn/stornext/d16/cmbco/pasiphae/eiribrat/Gaia/'
    readpath1  = directory #+ filename
    

    print(os.path.abspath(os.curdir),': Home directory')
    for i in range(3):
        os.chdir('..')
    
    print(os.path.abspath(os.curdir), ': Groups directory')
    path = os.chdir(directory)
    print(os.path.abspath(os.curdir), ': Gaia data directory')
    print('Read .csv data from Gaia using columns:', colnames)
    print('***************')
    start = time.time()
    for j in range(0,10,1):
        print('Run for the two last digits: {}{}'.format(j, 0))
        get_data('{}*{}{}.csv.gz'.format(readpath1, j, 0), '{}{}{}'.format(savepath, j, 0), colnames)

        #get_csv_data('{}*{}{}.csv.gz'.format(readpath1, j,0), '{}{}{}'.format(savepath, j,0)) # owl18
    
        end = time.time()
        print('------ Time used: {} s ------'.format(end-start))
    
    ##
    return None



"""
Call functions:
"""

N_side    = 128

#filename  = 'GaiaSource_999922404314639104_1000172126596665472.csv.gz'
#directory = '/mn/stornext/d16/cmbco/eirikgje/data/gaia/cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/'

#savepath  = '/mn/stornext/d16/cmbco/pasiphae/eiribrat/Gaia/'
#readpath1  = directory #+ filename
#readpath2  = directory #+ 'GaiaSource_999717001796824064_999922369954904960.csv.gz'


print('======================')
#start2 = time.time()
#start = time.time()
"""
##########################
# test for 2 random files:
get_csv_data('{}'.format(readpath1), '{}{}test'.format(savepath, 'test1'))
get_csv_data('{}'.format(readpath2), '{}{}test'.format(savepath, 'test2'))

fileMerger(savepath+'*ra', 'ra', savepath+'ra_test', 'float')   
fileMerger(savepath+'*dec', 'dec', savepath+'dec_test', 'float')
##########################
#"""

#end = time.time()
#print('Time: {} s'.format(end-start))


# generate HDF files from .csv files:
# -->
#call_get_data(0, ['ra','dec'])
#call_get_data(2, [])
#call_get_data(4, [])
#call_get_data(6, [])
#call_get_data(8, [])


"""        
end2 = time.time()
print('file generating time: {} sec'.format(end2-start2))
#"""

print(os.path.abspath(os.curdir), ': Current directory')

#########
# Merge parameter files into one parmeter file:
# -->
#fileMerger('ra', 'ra', 'RightAscension', 'float')  # savepath+param, name, savepath+out_file_name, data type
#fileMerger('ra_error', 'ra_error', 'RA_error', 'float')
#fileMerger('dec', 'dec', 'Declination', 'float')
#fileMerger('dec_error', 'dec_error', 'Declination_error', 'float')
#fileMerger('parallax', 'parallax', 'Parallax', 'float')
#fileMerger('parallax_error', 'parallax error', 'Parallax_error', 'float')
#fileMerger('mean_mag_G', 'mean_mag_G', 'Mean_Mag_G', 'float')
#fileMerger('mean_mag_BP', 'mean_mag_BP', 'Mean_Mag_BP', 'float')
#fileMerger('mean_mag_RP', 'mean_mag_RP', 'Mean_Mag_RP', 'float')
#fileMerger('source_id', 'source_id', 'Source_ID', 'int')
#fileMerger('random_index', 'random index', 'Random_index', 'int')

########

#end2 = time.time()
#print('Total time used: {} s'.format(end2-start2))
print('======================')



