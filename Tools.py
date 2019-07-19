import numpy as np
import matplotlib.pyplot as plt

import healpy as hp
import sys, os, glob, time
import h5py
import Datahandling as dh

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

import itertools as it
import numba as nb



"""
This program contain functions to calculate properties in DistanceMapping.py and Datahandling.py

"""


############
# params:
ZP_blue        = 25.3513881707
ZP_red         = 24.7619199882
ZP_green       = 25.6883657251




############
# Functions:
@nb.jit(nopython=True)
def Map(a, bins):
    """
    Create map of coordinates.
    """
    
    hist, b = np.histogram(a, bins)
    return hist, b


def DistError(dist):
    """
    Compute the distance error from the uncertainty in the parallax.
    Input: dist, array 
    """

    f   = h5py.File('Parallax_error.h5','r')
    err = np.asarray(f['parallax error'])
    f.close()

    #f2  = h5py.File('Parallax.h5', 'r')
    #p   = np.asarray(f2['parallax'])
    #f2.close()
    
    #err      = np.nan_to_num(err)
    p        = 1000./dist
    
    dist_err = err/(p**2)
    dist_err = np.nan_to_num(dist_err)
   
    #plt.histogram(dist_err, bins=100)
    #plt.show()

    return dist_err
    # End DistError


def IntegrateOvererror(mu, sigma, b2):
    """
    If distance uncertainty is over 100%, then integrate up to an appropiate
    bin and add there, moste likely a small contribution.
    Input: - mu, scalar, mean distance measured
           - sigma, scalar, error in distance
           - b, scalar, upper bin edge
    """
    
    N    = 1000
    y    = np.random.normal(mu, sigma, N)
    bins = np.arange(0, np.ceil(y.max()), N)#y.max()/N)
    dx   = (bins[1]-bins[0])
    
    c, x = Map(y, bins=bins)
    #print(len(bins), len(x), dx)    
    fx   = 1./(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2.*sigma**2))

    a    = 0
    for i in range(len(x)-1):
        if (x[i] < b2): # & (x[i] > b1)
            a += fx[i]*dx
    #print(a)
    return a
    

def IntegrateError(mu, sigma, edge):
    """
    Function for integrating over the distance uncertainty close to a bin edge.
    Input: - mu, scalar, mean distance measured
           - sigma, scalar, error in distance
           - edge, scalar, upper bin edge of the distance inteval in progress.
    """

    N    = 1000
    y    = np.random.normal(mu, sigma, N)
    bins = np.arange(0, np.ceil(y.max()), y.max()/N)
    dx   = abs(bins[1])- abs(bins[0])
    #print(bins)
    c, x = np.histogram(y, bins=bins)
    #print(len(bins), len(x))    
    fx   = 1./(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    
    a    = 0
    for i in range(len(x)-1):
        if x[i] <= edge:
            a += fx[i]*dx
    #print(a)
    return a
    # end Integrate


def AddIntensity(filter=None):
    """
    Add intensity to the pixels, bright stars give higher pixel values, faint stars lower pixel values
    Input: filter, optional, string, which passband filter to use, 'G', 'BP', 'RP' or mean. Default is None
    
    """
    

    if filter==None:
        print('Use no filters')
        return None

    if filter=='mean':
        print('Use all filters and get the mean magnitude')

        f1     = h5py.File('Mean_Mag_G.h5', 'r')
        mag_G  = np.asarray(f1['mean_mag_G'])
        f1.close()
        f2      = h5py.File('Mean_Mag_BP.h5', 'r')
        mag_BP  = np.asarray(f2['mean_mag_BP'])
        f2.close()
        f3      = h5py.File('Mean_Mag_RP.h5', 'r')
        mag_RP  = np.asarray(f3['mean_mag_RP'])
        f3.close()
    
        # Use zero point values for stars with unknown magnitude
        Gnan           = np.isnan(mag_G)
        BPnan          = np.isnan(mag_BP)
        RPnan          = np.isnan(mag_RP)        
        ifnanG         = np.where(Gnan==True)
        ifnanB         = np.where(BPnan==True)
        ifnanR         = np.where(RPnan==True)        
        mag_G[ifnanG]  = ZP_green
        mag_BP[ifnanB] = ZP_blue
        mag_RP[ifnanR] = ZP_red
    
        # Weights
        wG    = 1./(10**mag_G)
        wB    = 1./(10**mag_BP)
        wR    = 1./(10**mag_RP)
        wMean = (wG + wB + wR)/3.
        return wMean

    if filter=='G':
        print('Use green filter')
        f1     = h5py.File('Mean_Mag_G.h5', 'r')
        mag_G  = np.asarray(f1['mean_mag_G'])
        f1.close()

        Gnan          = np.isnan(mag_G)
        ifnanG        = np.where(Gnan==True)
        mag_G[ifnanG] = ZP_green
        
        wG            = 1./(10**mag_G)
        return wG

    if filter=='BP':
        print('Use blue filter')
        f1     = h5py.File('Mean_Mag_BP.h5', 'r')
        mag_B  = np.asarray(f1['mean_mag_BP'])
        f1.close()

        Bnan          = np.isnan(mag_B)
        ifnanB        = np.where(Bnan==True)
        mag_B[ifnanB] = ZP_blue
        
        w             = 1./(10**mag_B)
        return w

    if filter=='RP':
        print('Use red filter')
        f1     = h5py.File('Mean_Mag_RP.h5', 'r')
        mag  = np.asarray(f1['mean_mag_RP'])
        f1.close()

        Rnan = np.isnan(mag)
        ifnanR = np.where(Rnan==True)
        mag[ifnanR] = ZP_red
        
        w = 1./(10**mag)
        return w

    # End AddIntensity

@nb.jit(nopython=True)
def Parallax2dist(p):
    """
    Compute the distance to the stars in [pc] from the parallax angel in [mas]
    """
    p = p/1000.      # milli arcsec to arcsec
    return 1.0/p
    # end Parallax2dist  


def PlotDist_DistError():
    """
    Plot the relationship between distance and distance error
    """
    f1 = h5py.File('Distance.h5','r')
    d  = np.asarray(f1['distance'])
    f1.close()
   
    d  = np.nan_to_num(d)
    ii = np.nonzero(d)
    d  = d[ii]
    
    de = DistError(d)

    percentile = np.abs(de)/d
    
    plt.figure('Distance vs. Error')
    plt.loglog(np.abs(d), de, c='.r')
    plt.xlabel('Distance [pc]', size=14)
    plt.ylabel('Distance error [pc]', size=14)
    

    plt.figure('Distance vs percentile')
    plt.semilogx(np.abs(d), percentile, c='.r')
    plt.xlabel('Distance [pc]', size=14)
    plt.ylabel(r'$d/\delta d$', size=14)
    
    plt.show()
    
def NoDistance(dist=None, plot=False):
    """
    Make a unfiltered map of the stars with unknown distance. Return the non zero
    indices.
    Input: - dist, array, optional, array of distances
           - plot, bool,  if True make plot of the stars with no distance,
                          else no plot.
    """
    # dist ??
    if dist.all() == None:
        print('Read distance file')
        f1 = h5py.File('Distance.h5','r')
        d  = np.asarray(f1['distance'])
        f1.close()
        
        d    = nan_to_num(d)
        ind  = np.where(d==0)
        ii   = np.nonzero(d)
        dd   = d[ind]
        #dist = d[ii]
    
    else:
        dist = np.nan_to_num(dist)
        ind  = np.where(dist==0)
        dd   = dist[ind]
        ii   = np.nonzero(dist)
        #dist = dist[ii]

    # Plot ??
    if plot==True:
        Ns   = 1024
        Npix = hp.nside2npix(Ns)
        print('Make plot for Nside={}'.format(Ns))
        f    = h5py.File('SPC_Nside_1024.h5','r')
        p    = np.asarray(f1['distance'])
        f1.close()
            
        m, b = Map(p[ind], Npix)
        hp.mollview(m, coord=['C','G'], nest=False, title='Stars with unknown distance, Nside={}'.format(Ns), unit='Nstars')
        plt.show()
    # end plot if
    return ii

    # end NoDistance



def Create_3Dmap(Max_r, xy=False, xz=False, yz=False):
    """
    Create a 3D map in xyz coordinates, and plane maps in input coordinates.
    Input: - x, bool, default = False
           - y, ------- " ----------
           - z, ------- " ----------
    """
    
    f1   = h5py.File('RightAscension.h5','r')
    ra   = np.asarray(f1['ra'])
    f1.close()

    f2   = h5py.File('Declination.h5','r')
    dec  = np.asarray(f2['dec'])
    f2.close()

    f3   = h5py.File('Distance.h5','r')
    dist = np.asarray(f3['distance'])
    f3.close()
    print('Data is loaded, get stars up to {} kpc into arrays'.format(Max_r))

    dist = np.nan_to_num(dist)
    ind  = np.where((dist <= Max_r) & (dist > 0)) 

    ra   = ra[ind]
    dec  = dec[ind]
    dist = dist[ind]
    dec  = np.pi/2. - dec
    print('Number of stars:', len(dist))

    xyz_coord = hp.pixelfunc.ang2vec(dec, ra)
    xyz       = np.zeros((len(ra), 3))
    #print(np.shape(xyz_coord), np.shape(xyz))

    for i in range(3):
        xyz[:,i] = dist*xyz_coord[:,i]/1000.

    print('Create 3D map')
    fig = plt.figure('xyz coordinates')
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c='y', s=0.01)
    ax.set_xlabel('kpc')
    ax.set_ylabel('kpc')
    ax.set_zlabel('kpc')

    if (xy == True):
        print('Plot xy plane')
        plt.figure('xy plane')
        plt.scatter(xyz[:,0], xyz[:,1], c='y', s=0.1)
        plt.xlabel('kpc')
        plt.ylabel('kpc')

    if (xz == True):
        plt.show('Plot xz plane')
        plt.figure('xy plane')
        plt.scatter(xyz[:,0], xyz[:,2], c='y', s=0.1)
        plt.xlabel('kpc')
        plt.ylabel('kpc')

    if (yz == True):
        print('Plot yz plane')
        plt.figure('xy plane')
        plt.scatter(xyz[:,1], xyz[:,2], c='y', s=0.1)
        plt.xlabel('kpc')
        plt.ylabel('kpc')

    plt.show()
    
    
#Create_3Dmap(1000)
#t1 = time.time()
#IntegrateError(14790.0, 14790.0*0.2, 15000)
#IntegrateOvererror(14796145.0, 14796145.0*2, 1e7)
#t2 = time.time()
#print(t2-t1)
#plt.show()
