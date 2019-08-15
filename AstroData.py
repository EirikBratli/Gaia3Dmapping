"""
This program make files with astro physical data form HDF 5 files.
It contain the functions: PixelCoord() and Positionfunc(), first function 
find the Healpix coordinates of the stars/objects write the Healpix 
coordinates to a file, it also returns the array containing the Healpix
coordinates. The second function finds the distance to the objects using 
parallax, then it writes the distances to a file.
"""

import h5py
import numpy as np
import params, Tools


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
    
    
    # Write Healpix coordinate file    
    Nstars = len(ra)
    pixpos = np.zeros(Nstars)
    pixpos = hp.pixelfunc.ang2pix(Nside, dec, ra)
    
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

    dist      = Tools.Parallax2dist(parallax)
   
    a = dist_file.split('/')
    path = '/'
    for i in range(1, len(a)-1):
        path += a[i]+'/'
    if len(a) <= 1:
        path = ''
    else:
        path = path

    # Write distance file    
    f = h5py.File(path+'Distance.h5', 'w')
    f.create_dataset('distance', data=dist.astype(dtype=np.float32))
    f.close()
    
    # end Positionfunc      



##################
N_side = 128
#Positionsfunc('Parallax.h5')
#PixelCoord(N_side, 'RightAscension.h5', 'Declination.h5', '')
