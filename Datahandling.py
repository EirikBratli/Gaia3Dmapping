from astropy.io import fits
import pandas as pd
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import glob, sys, os
import h5py
import time

# EIRIK: It seems like some of these data are for file input/output while
# others are for handling the actual astrophysical data. I would make a clean
# separation between those two concepts and make two modules for it.

"""
- Read in .csv data files fo Gaia DR2 and convert relevant columns into .fits files.
  Run in 5 batches of each last digit of .csv file name (even numbers). For each batch
  loop over the second to last digit in file name. This does not use too much memory.
  Get 25 files per parameter. Need to merge the parameter files. Note the order the files 
  are read in.

- Create pixel map form position arrays read from the position files, RightAscension.h5 
  and Declination.h5. Returns a array with pixel coordinates for each star and write a 
  file containing the same array.
  

- Get xyz coordinates and distance. Calculate the distance to the stars using the parallax
  angel. (Also find the xyz position of the stars in a euclidian coordinat system using the 
  distance and coordinate transformation function in healpy.)
""" 

# EIRIK: In general, using uppercase letters for constants is 'best practice'.
# Further, I would recommend putting all constants in a separate module and
# then import that module. Global variables are in general frowned upon in
# Python.
# Convertion factors:
deg2rad = np.pi/180.0
unseen  = -1.6375e+30



def get_csv_data(read_path, save_path):
    """
    Read data from .csv file and write it to .h5 files. Sending in parts of total data files.
    
    """
    path = read_path #os.path.join(read_path, 'GaiaSource_999922404314639104_1000172126596665472.csv')
    print('read path:', path)
    a = path.split('.')
    #b = save_path.split('/')
    #save_path = b[-1]
    print('Save path:', save_path)
    
    #print(len(os.listdir(path)))
    
    # EIRIK: Every string has an .endswith() function that you should use
    # instead of this method.
    if a[-2] == 'csv':
         print('Read .csv files')
         datafiles = glob.glob(path)
         print(datafiles[-5:])
         print(datafiles[:5])
         datafiles.sort()
         print('Number of data files: {}'.format(len(datafiles)))
         print(datafiles[-5:])
         print(datafiles[:5])
    else:
        print('There are no .csv files in the path folder.')
        sys.exit()
    

    #####
    if len(datafiles) > 0:
        
        print('Usefull columns in datafiles: designation, random_index, ra, ra_error,dec, dec_error, parallax, parallax_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag')
        
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
            #EIRIK: This is an ok thing to do I suppose, though slightly more
            # elegant if we used something in the np library. What about
            # genfromtxt?
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
    
        ra  = ra*deg2rad
        dec = dec*deg2rad 
        
        # write parameter files
        #""" 
        # EIRIK: This should be done in a for loop, using a dictionary or
        # similar for the things that vary between each loop. No real need to
        # have a new name for the file object variable either (just hdf is
        # fine, not hdf0, hdf1, hdf2...)
        hdf0 = h5py.File('{}_source_id.h5'.format(save_path), 'w')
        hdf0.create_dataset('source_id', data=source_id.astype(int))  
        hdf0.close()
        hdf1 = h5py.File('{}_random_index.h5'.format(save_path), 'w')
        hdf1.create_dataset('random index', data=random_index.astype(int))
        hdf1.close()
        #"""
        hdf2 = h5py.File('{}_ra.h5'.format(save_path), 'w')
        hdf2.create_dataset('ra', data=ra.astype(dtype=np.float32))
        hdf2.close()
        #"""
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
        #"""

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
    #print(len(files))
    #"""
    for file in files:
        read_file = h5py.File('{}'.format(file), 'r')
        dt        = np.asarray(read_file['{}'.format(name)])
        read_file.close()
       
        Data      = np.append(Data, dt)
        print(file, len(dt))
        # delete input files
        if os.path.isfile(file):
            print('Delete file: {}'.format(file))
            os.remove(file)
        else:
            print('Error: {} is not found'.format(file))
    
    print(len(Data))
    #"""
    # Write new file:
    outdata = h5py.File('{}.h5'.format(out_file_name), 'w')

    if type == 'int':
        outdata.create_dataset('{}'.format(name), data=Data.astype(int))

    elif type=='float':
        outdata.create_dataset('{}'.format(name), data=Data.astype(dtype=np.float32))

    elif type=='str':
        outdata.create_dataset('{}'.format(name), data=Data.astype(str))

    outdata.close()
    #"""
    # End fileMerger


def PixelCoord(Nside, ra_file, dec_file, save_path):
    """
    Make file with Healpix coordinates from ra and dec files.
    """
    f1   = h5py.File(ra_file, 'r')
    ra   = np.asarray(f1['ra'])
    f1.close
    f2   = h5py.File(dec_file, 'r')
    dec  = np.asarray(f2['dec'])    # Not healpix coordinates (0, pi), have (-pi/2, pi/2)
    f2.close()

    # Transform dec to healpix 
    dec = np.pi/2.0 - dec           
    
    
    # Write to files:
    # Healpix coordinate file       
    Nstars = len(ra)
    pixpos = np.zeros(Nstars)
    #for i in range(Nstars):
    pixpos = hp.pixelfunc.ang2pix(Nside, dec, ra)
    
    #print(pixpos)
    
    pixcoord = h5py.File('{}SPC_Nside_{}.h5'.format(save_path, Nside), 'w')
    pixcoord.create_dataset('Healpix coordinates', data=pixpos.astype(int))
    pixcoord.close()

    return pixpos
    # end PixelCoord


def Positionsfunc(dist_file):
    print('Get the distance to the star')
               
    hdf3     = h5py.File(dist_file)
    parallax  = np.asarray(hdf3['parallax'])       
    hdf3.close()

    dist      = Parallax2dist(parallax)
   
    a = dist_file.split('/')
    print(a, len(a))
    path = '/'
    for i in range(1, len(a)-1):
        path += a[i]+'/'
    if len(a) <= 1:
        path = ''
    else:
        path = path
    print(path)

    # Write distance file    
    f = h5py.File(path+'Distance.h5', 'w')
    f.create_dataset('distance', data=dist.astype(dtype=np.float32))
    f.close()
    
    # end Positionfunc           



def Parallax2dist(p):
    """
    Compute the distance to the stars in [pc] from the parallax angel in [mas]
    """
    p = p/1000.      # milli arcsec to arcsec
    return 1.0/p
    # end Parallax2dist           




#####################
"""
Call functions:
"""

N_side    = 128

filename  = 'GaiaSource_999922404314639104_1000172126596665472.csv.gz'
directory = '/mn/stornext/d16/cmbco/eirikgje/data/gaia/cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/'

savepath  = '/mn/stornext/d16/cmbco/pasiphae/eiribrat/Gaia/'
readpath1  = directory #+ filename
readpath2  = directory #+ 'GaiaSource_999717001796824064_999922369954904960.csv.gz'


print('======================')
#"""
#print(os.path.abspath(os.curdir),': Eiriks directory')
#for i in range(3):
#    os.chdir('..')
    
#print(os.path.abspath(os.curdir), ': Groups directory')
#path = os.chdir(directory)
#print(os.path.abspath(os.curdir), ': Gaia data directory')
#"""
#print('----')
#print(readpath1)
#print(savepath)
#"""
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


#for j in range(0,10,1):
#    print('Run for the two last digits: {}{}'.format(j, 0))
#    get_csv_data('{}*{}{}.csv.gz'.format(readpath1, j,0), '{}{}{}'.format(savepath, j,0)) # owl18
#    #get_csv_data('{}*{}{}.csv.gz'.format(readpath1, j,2), '{}{}{}'.format(savepath, j,2)) # owl20
#    #get_csv_data('{}*{}{}.csv.gz'.format(readpath1, j,4), '{}{}{}'.format(savepath, j,4)) # owl19
#    #get_csv_data('{}*{}{}.csv.gz'.format(readpath1, j,6), '{}{}{}'.format(savepath, j,6)) # owl21
#    get_csv_data('{}*{}{}.csv.gz'.format(readpath1, j,8), '{}{}{}'.format(savepath, j,8)) # owl22
#    end = time.time()
#    print('------ Time used: {} s ------'.format(end-start))
    


"""        
end2 = time.time()
print('file generating time: {} sec'.format(end2-start2))
#"""

print(os.path.abspath(os.curdir), ': Current directory')

#########
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


##########
#Positionsfunc('Parallax.h5')
#PixelCoord(N_side, 'RightAscension.h5', 'Declination.h5', '')

#########

#end2 = time.time()
#print('Total time used: {} s'.format(end2-start2))
print('======================')



