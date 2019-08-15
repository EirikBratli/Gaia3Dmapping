import numpy as np
import matplotlib.pyplot as plt

import healpy as hp
import sys, os, glob, time
import h5py
import params

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numba as nb



"""
This program contain functions to calculate properties in DistanceMapping.py and Datahandling.py

"""


############
# Functions:
@nb.jit(nopython=True)
def Map(a, bins):
    """
    Create map of coordinates.
    """
    
    hist, b = np.histogram(a, bins)
    return hist, b


def Find_Nside(Nside, Nrad):
    """
    Find the most appropriate Nside as function of number of stars in the Current
    bin.
    Input: - Nside, list or scalar, if scalar use the given Nside. If list use the most approprate Nside
           - Nrad, int,             the number of stars in the current bin

    Returns: The best fitting Nside value
             The Npix value
    """

    if isinstance(Nside, list) == True:
        # Let the number of stars in a bin give the Nside
        Nside.sort()
        if len(Nside) == 2:
            if len(rad) > 5e7:
                Ns = max(Nside)
            else:
                Ns = min(Nside)


        elif len(Nside) == 3:
            if len(rad) > 1e8:
                Ns = max(Nside)

            elif len(rad) < 1e7:
                Ns = min(Nside)

            else:
                Ns = Nside[1]

        elif len(Nside) > 3:
            print('Too many Nsides, max 3')
            sys.exit()

        ###
    else:
        # use one Nside the scalar
        Ns = Nside

    Npix = hp.nside2npix(Ns)
    return Ns, Npix


def PlotMaps(pix, Ns, Npix, w, rad, Bins, i, filter=None, h=False):
    """
    Plot funciton for Distance mapping
    """
    if filter == None:
        logmap, b = Map(pix, Npix)
    else:
        pos_map, b = np.histogram(pix, Npix, weights=w)
        logmap = np.log(pos_map)
        logmap[np.isinf(logmap)] = -60

    hp.mollview(logmap, coord=['C','G'],nest=False, title='Stars form {} pc to {} pc, Nside={}'.format(Bins[i], Bins[i+1], Ns), unit='Nstars')
    plt.savefig('Figures/Maplayer{}_Nside{}.png'.format(i,Ns))

    if h == True:
        plt.figure('Histogram {}'.format(i))
        plt.hist(rad, bins=50)
        plt.xlabel('pc')
        plt.ylabel('Number of stars')
        plt.savefig('Figures/MaplayerHist{}_Nside{}.png'.format(i,Ns))




def DistError(dist):
    """
    Compute the distance error from the uncertainty in the parallax.
    Input: dist, array 
    """

    f   = h5py.File('Parallax_error.h5','r')
    err = np.asarray(f['parallax error'])
    f.close()

    p        = 1000./dist
    
    dist_err = err/(p**2)
    dist_err = np.nan_to_num(dist_err)
   
    return dist_err
    # End DistError


def ApplyErrorNmaps(Nmaps, i, ind, Bins, pixcoord, rad_dist, dist_err, percentile, weight, uu_ind, over_ind, OverError=False):
    """
    Function for using Nmaps in distance mapping
    """
    
    l_ind, u_ind    = [],[]
    eps             = Bins[i+1]*0.01
        
    pixel = pixcoord[ind]
    per   = percentile[ind]
    w     = weight[ind]
    c     = 0
                
    iup = np.where((percentile > 0.01) & (percentile < 1) & (rad_dist > Bins[i+1]-eps) & (rad_dist <= Bins[i+1]))
    ilo = np.where((percentile > 0.01) & (percentile < 1) & (rad_dist <= Bins[i] +eps) & (rad_dist > Bins[i]))
                
                
    print('-->')
    print('{} stars close to lower edge {}pc with large error'.format(len(ilo[0]),Bins[i]))
    print('{} stars close to upper edge {}pc with large error'.format(len(iup[0]),Bins[i+1]))
    print(len(pixel), len(w))

    weight, wuu = ErrorHandling(rad_dist, dist_err, percentile, Bins, weight, uu_ind, ilo, iup, i)
                
    if len(uu_ind) != 0:
        pixel = np.concatenate((pixel, pixcoord[uu_ind]), axis=0)
        w     = np.concatenate((weight[ind], wuu), axis=0)
        rad   = np.concatenate((rad_dist[ind], rad_dist[uu_ind]), axis=0)

    else:
        pixel = pixel
        w     = weight[ind]
        rad   = rad_dist[ind]
    #end if
    
    if (i == Nmaps-1) and (OverError==True):
        w0, p, r = OverError(over_ind, Bins, rad_dist, dist_err, weight, pixcoord)
        pixel = np.concatenate((pixel, p), axis=0)
        w     = np.concatenate((w, w0), axis=0)
        rad   = np.concatenate((rad, r), axis=0)
        
    # end apply error if test
    return w, pixel, rad, iup[0]
    

def ApplyErrorBinSeq(Bin_seq, i, ind, pixcoord, rad_dist, dist_err, percentile, weight, uu_ind, over_ind, cutoff, OverError=False):
    """
    Function when using Bin_seq in distance mapping
    """

    l_ind, u_ind    = [],[]
    eps             = Bin_seq[i+1]*0.01
            
    pix = pixcoord[ind]
    per = percentile[ind]
    w   = weight[ind]
    c   = 0
                    
    iup = np.where((percentile > 0.01) & (percentile < 1) & (rad_dist > Bin_seq[i+1]-eps) & (rad_dist <= Bin_seq[i+1]))
    ilo = np.where((percentile > 0.01) & (percentile < 1) & (rad_dist <= Bin_seq[i] +eps) & (rad_dist > Bin_seq[i]))
                    
    print('-->')
    print('{} stars close to lower edge {}pc with large error'.format(len(ilo[0]),Bin_seq[i]))
    print('{} stars close to upper edge {}pc with large error'.format(len(iup[0]),Bin_seq[i+1]))
                    
    # Update the weights by integrating the distance error near the upper and lower bin edges 
    weight, wuu = ErrorHandling(rad_dist, dist_err, percentile, Bin_seq, weight, uu_ind, ilo, iup, i)

    if len(uu_ind) != 0:
        pix   = np.concatenate((pix, pixcoord[uu_ind]), axis=0)
        w     = np.concatenate((weight[ind], wuu), axis=0)
        rad   = np.concatenate((rad_dist[ind], rad_dist[uu_ind]), axis=0)

    else:
        pix   = pix
        w     = weight[ind]
        rad   = rad_dist[ind]
    #end if

    # Over errors:
    if (i == cutoff-1) and (OverError==True):
        w0, p, r = OverError(over_ind, Bin_seq, rad_dist, dist_err, weight, pixcoord)
        
        pix   = np.concatenate((pixel, p), axis=0)
        w     = np.concatenate((w, w0), axis=0)
        rad   = np.concatenate((rad, r), axis=0)
    # end if

    return pix, w, rad, iup[0]                



def ErrorHandling(rad_dist, dist_err, percentile, Bins, weight, uu_ind, ilo, iup, i):
    """
    Compute the contribution of the stellar weight to the current bin.
    Input: - Radial distance array
           - array with distance error
           - array with the percentile of the distance and error
           - Array with the bin edges
           - Array with the weights
           - seq. with the indices to be added from previous upper bin edge
           - seq. with the indices to be added form the previous lower bin edge
           - seq. with indices to be looped over, lower bin edge
           - seq. with indices to be looped over, upper bin edge
           - integer, index of the current bin
    
    Returns: Array with the weights
             Array with the weights to be added to the next bin
             Array with the weights contrinbution going into the previous bin ??
    """
    if len(ilo[0]) == 0:
        wll = np.empty(0)
    else:
        print('compute error for lower bin edge')
        c = 0
        for k,j in enumerate(ilo[0]):
            a         = IntegrateError(rad_dist[j], dist_err[j], Bins[i])
            weight[j] = weight[j]*(1-a)
            #wll       = weight[ll_ind]*a
            c += 1
        print('Number of contributions on lower bin edge: {}'.format(c))
    # end lower bin loop

    if len(iup[0]) == 0:
        wuu = np.empty(0)
    else:
        print('Compute error for upper bin edge')
        c = 0
        for k,j in enumerate(iup[0]):
            c += 1
            # integrate error
            a         = IntegrateError(rad_dist[j], dist_err[j], Bins[i+1])
            weight[j] = weight[j]*a
            wuu       = weight[uu_ind]*(1-a)
        print('Number of contributions on upper bin edge: {}'.format(c))
    ## end upper bin loop

    return weight, wuu #, wll


def OverError(over_ind, Bins, rad_dist, dist_err, weight, pixcoord):
    """
    Compute the contribution of the over errors for the last bin.
    Input: - over_ind, seq.   Sequence with the indices of large error
           - Bins, seq.       List of bin edges
           - rad_dist, seq    Array with the radial distances
           - dist_err, seq    Array with the distance errors
           - weights, seq     Array with the weights
           - pixcoord, seq    Array with pixel coordinates
    Returns: lists with update weights
             list with the pixel coordinates for the over error stars
             list with the distances
    """
    w0 = []
    p  = []
    r  = []
    c  = 0
    for k in over_ind:
        if (rad_dist[k] < 100*Bins[-1]) and rad_dist[k] > Bins[-1]:
            a         = IntegrateOvererror(rad_dist[k], dist_err[k], Bins[-1]) # why slow??
            weight[k] = a*weight[k]
            c += 1
            r.append(rad_dist[k])
            w0.append(weight[k])
            p.append(pixcoord[k])
        #end if
    #end for loop
    print('Add {} number of stars to last bin with large uncertainty'.format(c))
    return w0, p, r


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
    c, x = np.histogram(y, bins=bins)
    fx   = 1./(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    
    a    = 0
    for i in range(len(x)-1):
        if x[i] <= edge:
            a += fx[i]*dx
    return a
    # end Integrate


def IntegrateOvererror(mu, sigma, b):
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
    fx   = 1./(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2.*sigma**2))

    a    = 0
    for i in range(len(x)-1):
        if (x[i] < b): 
            a += fx[i]*dx
    return a


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
        mag_G[ifnanG]  = params.ZP_GREEN
        mag_BP[ifnanB] = params.ZP_BLUE
        mag_RP[ifnanR] = params.ZP_RED
    
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
        mag_G[ifnanG] = params.ZP_GREEN
        
        wG            = 1./(10**mag_G)
        return wG

    if filter=='BP':
        print('Use blue filter')
        f1     = h5py.File('Mean_Mag_BP.h5', 'r')
        mag_B  = np.asarray(f1['mean_mag_BP'])
        f1.close()

        Bnan          = np.isnan(mag_B)
        ifnanB        = np.where(Bnan==True)
        mag_B[ifnanB] = params.ZP_BLUE
        
        w             = 1./(10**mag_B)
        return w

    if filter=='RP':
        print('Use red filter')
        f1     = h5py.File('Mean_Mag_RP.h5', 'r')
        mag  = np.asarray(f1['mean_mag_RP'])
        f1.close()

        Rnan = np.isnan(mag)
        ifnanR = np.where(Rnan==True)
        mag[ifnanR] = params.ZP_RED
        
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


def PlotDist_DistError(dist=None):
    """
    Plot the relationship between distance and distance error
    """
    if dist is None:
        f1 = h5py.File('Distance.h5','r')
        d  = np.asarray(f1['distance'])
        f1.close()

    else:
        d  = dist
   
    print('plotting')
    
    de = DistError(d)
    d  = np.nan_to_num(d)
    ii = np.nonzero(d)
    d  = d[ii]
    de = de[ii]
    d  = d[::10]
    de = de[::10]

    percentile = np.abs(de/d)

    plt.figure('Distance vs. Error')
    plt.loglog(np.abs(d), np.abs(de), '.r')
    plt.xlabel('Distance [pc]', size=14)
    plt.ylabel('Distance error [pc]', size=14)
    plt.savefig('Figures2/Dist_vs_err.png')
    
    plt.figure('Distance vs percentile')
    plt.loglog(np.abs(d), percentile, '.r')
    plt.xlabel('Distance [pc]', size=14)
    plt.ylabel(r'$d/\delta d$', size=14)
    plt.savefig('Figures2/dist_percentile.png')
    plt.show()

    
def NoDistance(dist, plot=False):
    """
    Make a unfiltered map of the stars with unknown distance. Return the non zero
    indices.
    Input: - dist, array, optional, array of distances
           - plot, bool,  if True make plot of the stars with no distance,
                          else no plot.

    Return: ii, indices where dist = unknown.
    """

    dist = np.nan_to_num(dist)
    ind  = np.where(dist==0)
    dd   = dist[ind]
    ii   = np.nonzero(dist)

    # Plot
    if plot==True:
        Ns   = 512
        Npix = hp.nside2npix(Ns)
        print('Make plot for Nside={}'.format(Ns))
        f    = h5py.File('SPC_Nside_1024.h5','r')
        p    = np.asarray(f['Healpix coordinates'])
        f.close()
            
        m, b = Map(p[ind], Npix)
        hp.mollview(m, coord=['C','G'], nest=False, title='Stars with unknown distance, Nside={}'.format(Ns), unit='Nstars')
        #plt.show()
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
    
