import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import sys, os, glob, time
import h5py
import AstroData as ad

from matplotlib import cm
from pathlib import Path

import Tools
import numba as nb
import params

"""
Make distribution of stars within different distances. So can make a 3D map of
the distributions.
"""



def testNside(lower,upper):
    """
    Test different Nsides resolutions of the maps, see what is best. 
    Input: - lower, int-scalar,
           - upper, int-scalar,    Both must follow 2**p.
    """
    if lower==False and upper==False:
        print('No input! Need lower and upper power of 2, in Nside=2**p')
        sys.exit()
    elif lower==False:
        print('No lower limit, type in first argument. Nside=2**p')
        sys.exit()
    elif upper==False:
        print('No upper limit, type in second argument. Nside=2**p')
        sys.exit()
    else:
        pass
        

    if lower<0 or upper<0:
        print('Power must be positive')
        sys.exit()
    elif lower>=12 or upper>12:
        print('Power too high, max p=12. Will use too long time to generate maps')
    else:
        pass

    savepath = ''
    f1 = 'RightAscension.h5'
    f2 = 'Declination.h5'

    for p in range(lower,upper+1):
        t1 = time.time()
        Ns = 2**p
        print('Nside={}'.format(Ns))
        
        # check if files exist:
        fil = Path("SPC_Nside_{}.h5".format(Ns))
        print(fil)
        if fil.exists():
            # read the pixel file:
            print('Read coordinate file for Nside = {}'.format(Ns))
            f = h5py.File(fil, 'r')
            pixpos = np.asarray(f['Healpix coordinates'])
            f.close()
        
        else:    
            print('Create coordinate file for Nside = {}'.format(Ns))
            pixpos = ad.PixelCoord(Ns, savepath+'RightAscension.h5', savepath+'Declination.h5', savepath)

        Npix = hp.nside2npix(Ns)
        map, b = Tools.Map(pixpos, Npix)
        # plot
        hp.mollview(map, coord=['C','G'], nest=False, title='All stars with Nside={}'.format(Ns), unit='N_stars')
        plt.savefig('Figures2/AllStars_diff_Nside{}.png'.format(Ns))

        t2 = time.time()
        print('Time used for Nside {} is: {}'.format(Ns, t2-t1))
        print('_______________')

    #####    

    plt.show()
    # end testNside


def AllStars(Nside, filter=None):
    """
    Plot all stars in one map. And make a distribution of the distance to the stars.
    Input: Nside, filter
    """

    print('Plot all stars in map with filter = {}'.format(filter))

    f       = h5py.File('SPC_Nside_{}.h5'.format(Nside), 'r')
    pixcoord = np.asarray(f['Healpix coordinates'])
    f.close()


    Npix = hp.nside2npix(Nside)
    print('Plot all stars with Nside=',Nside)
    
    # Find a way to add intensity
    if filter != None:
        # Add intensity. With filters, Green, Blue, Red and their mean:
        print('Include magnitude as weights, with green, blue, red and mean')
        w  = Tools.AddIntensity(filter)
        
        Map, b = np.histogram(pixcoord, Npix, weights=w)
        print('Map with {} filter'.format(filter))
        logmap = np.log(Map)
        logmap[np.isinf(logmap)] = -60
        hp.mollview(logmap, coord=['C','G'], nest=False, title='All stars in {} filter, Nside = {}'.format(filter, Nside), unit='Nstars')
        plt.savefig('Figures2/AllStars_{}_Nside{}.png'.format(filter, Nside))
   
    else:
        # Map of all stars, No filters
        print('No filters') 
        m, b    = Tools.Map(pixcoord, Npix)    
        hp.mollview(m, coord=['C','G'], nest=False, title='All stars, Nside = {}'.format(Nside), unit='Nstars')
        plt.savefig('Figures2/AllStars_Nside{}.png'.format(Nside))

    
    plt.show()
    # End AllStars


def AllDistances(Bins, xscale='linear', yscale='linear'):
    """
    Make a histogram of the stellar distances of all stars. 
    Input: - list or scalar with bins, 
           - scale of x axis and y axis; 'linear', 'log', etc
    """
    
    f1       = h5py.File('Distance.h5','r')
    dist     = np.asarray(f1['distance']) 
    f1.close()
    
    dist     = np.nan_to_num(dist)
    rad_dist = np.abs(dist)
    i        = np.nonzero(rad_dist)
    rad_dist = rad_dist[i]
    print('Closest object at: {}pc most distant object at {}pc'.format(min(rad_dist), max(rad_dist)))
    
    # Plot histogram of the distancec distribution
    plt.figure('Distance distribution')
    plt.hist(rad_dist, bins=Bins)
    plt.gca().set_xscale(xscale)
    plt.gca().set_yscale(yscale)
    plt.xlabel('pc')
    plt.ylabel('Number of stars')
    plt.savefig('Figures2/Distance_distribution_mw.png')

    plt.show()
    # End AllDistances



def IntensityMapping(Nside, filter=None):
    """
    Distribute stars after their magnitude. Find which stars that have a magnitude
    within a range and plot the star in each magnitude range. Plot also weighted with 
    respect to magnitude
    Input: - Nside,  scalar
           - filter, string, either 'mean', 'G', 'BP', 'RP'.
    """
    
    f1       = h5py.File('SPC_Nside_{}.h5'.format(Nside), 'r')
    pixcoord = np.asarray(f1['Healpix coordinates'])
    f1.close()
    
    if filter != None:
        f3   = h5py.File('Distance.h5','r')
        dist = np.asarray(f3['distance'])
        f3.close()
        dist = np.nan_to_num(dist)
        ii   = np.nonzero(dist)
        rad  = np.abs(dist[ii])
    else:
        pass
    
    ####
    w    = Tools.AddIntensity(filter)
    if filter == None:
        f2 = h5py.File('Mean_Mag_G.h5','r')
        Mag = np.asarray(f2['mean_mag_G'])
        f2.close()
        
        n          = np.isnan(Mag)        
        ifnan      = np.where(n==True) 
        Mag[ifnan] = ZP_green
    else:
        Mag  = -np.log10(w)

    Npix = hp.nside2npix(Nside)
    N    = len(Mag)

    Bins = [26,20,15,10,5, np.min(Mag)*0.9]
    ind0 = np.where((Mag <= min(Bins))) # Check for stars with magnitude stronger than 0
    
    for i in range(len(Bins)-1):
        print('Magnitude range: {} to {}'.format(Bins[i], Bins[i+1]))

        ind = np.where((Mag < Bins[i]) & (Mag >= Bins[i+1]))
        if filter == None:
            m, b = Tools.Map(pixcoord[ind], Npix)
            logmap = np.log10(m)
            logmap[np.isinf(logmap)] = 0
            hp.mollview(logmap, coord=['C','G'], nest=False, title='Magnitude form {} mag to {} mag'.format(Bins[i], Bins[i+1]), unit='Nstars')
            plt.savefig('Figures2/Mag_dist_Nside{}.png'.format(Nside))

        if filter != None:
            Map, b = np.histogram(pixcoord[ind], Npix, weights=w[ind])
            logMap = np.log10(Map)
            logMap[np.isinf(logMap)] = -Bins[i]
            hp.mollview(logMap, coord=['C','G'], nest=False, title='Weighted for mag {} to {}'.format(Bins[i], Bins[i+1]), unit='Nstars')
            plt.savefig('Figures2/wMag_{}_Nside{}.png'.format(filter, Nside))
        
            plt.figure('hist {}'.format(i-1))
            plt.hist(rad[ind], 100)
            plt.xlabel('pc')
            plt.gca().set_xscale('log')
            plt.gca().set_yscale('log')
            plt.savefig('Figures2/mag_distribution_{}.png'.format(filter))
    
    plt.show()
    # End IntensityMapping

    

def DistributeDistance(Nside, Nmaps=None, Rmax=5e4, Rmin=0.1, Bin_seq=None, cutoff=None, filter=None, h=False, OverError=False):
    """
    Distribute distance of the stars, also apply the sky coordinate. 
    Input: - Nside,             int-scalar or list. Not array
           - Nmaps, optional,   Number of maps to generate
           - Rmax,              scalar, distance [pc] to generate maps up to, start from zero, default = 5e4 pc
                                used with Nmaps
           - Rmin,              scalar, distance [pc] to generate maps up to, start from zero, default = 0.1 pc
                                used with Nmaps
           - Bin_seq, optional, list of bin edges, manually made 
           - cutoff, optional,  int-scalar, upper limit of bin edge to iterate up to. Used with Bin_seq
           - filter, optional,  str, either 'mean', 'G', 'B', 'R'. Used in call for AddIntensity() function, 
                                default is 'None'   
           - h, optional,       bool, if True create histograms of the stellar distribution of distance. Else maps
           - OverError, bool    If True also apply the errors for stars with over error, percentile > 1. Else no 
                                computation ov over error
                            
    """ 
    if (Bin_seq==None) and (Nmaps==None):
        """
        No bins and no Nmaps. Abort
        """
        print('No bin sequence and no Nmaps as input argument.')
        sys.exit()

   
    # Read files
    f1       = h5py.File('Distance.h5', 'r')
    dist     = np.asarray(f1['distance'])
    f1.close()

    dist_err = Tools.DistError(dist)
    
    # check for nan values in dist:
    ii          = Tools.NoDistance(dist)
    dist        = dist[ii]
    dist_err    = dist_err[ii]
    print('Length of data:', len(dist))

    # radial distance
    rad_dist = np.abs(dist)
    Nside_old = 0  
    
    # Distance percentile:
    percentile = dist_err/rad_dist
    percentile = np.nan_to_num(percentile)
    
    # Add intesity:
    if filter != None:
        print('Load weights')
        weight = Tools.AddIntensity(filter)
        weight = weight[ii]
    else:
        print('Use no filter')
        weight = np.ones(len(rad_dist))
    
    # Test input parameters
    if (Bin_seq != None) and (Nmaps != None):
        """ 
        Have both bin seq. and Nmaps, then use Nmaps as default. 
        """

        print('Input both Bin_seq and Nmaps. Use Nmaps.')
        Bin_seq = None


    if Bin_seq == None:
        """
        Use Nmaps, no Bin sequence input
        Nmaps, Rmax, Rmin, rad_dist, dist_err, percentile, weight, ii
        """
        print('Use {} numbers of maps to map the star distribution'.format(Nmaps))
        size = (Rmax-Rmin)/Nmaps
        Bins = np.arange(Rmin, Rmax+10, size)
        print('Bin size: {}, bin edges:'.format(size))
        print(Bins)

        inte_ind       = np.where(percentile > 0.01)
        uu_ind, ll_ind = [], []
        
        over_ind       = np.where((percentile > 1) & (percentile < 10) & (rad_dist < 100*Bins[-1]) & (rad_dist > Bins[-1]))
        over_ind       = over_ind[0]
        
        OverUncer      = percentile[over_ind]

        # Loop over the the number of maps
        for i in range(Nmaps):

            print('=============')
            ind = np.where((rad_dist > Bins[i]) & (rad_dist <= Bins[i+1]))
            print('Bin number {}'.format(i))
            rad = rad_dist[ind]
           
            # Find appropriate Nside
            Ns, Npix = Tools.Find_Nside(Nside, len(rad))             
            print('Use Nside:',Nside)

            # load pix coordinate for the given Nside
            if Ns == Nside_old:
                pass
            else:
                # read file
                print('read file for Nside {}'.format(Ns))
                f2       = h5py.File('SPC_Nside_{}.h5'.format(Ns),'r')
                pixcoord = np.asarray(f2['Healpix coordinates'])
                f2.close()
                pixcoord = pixcoord[ii]

            ####
            # Apply error:
            if filter == None:
                
                pass
            
            else:
                w, pixel, rad, iup = Tools.ApplyErrorNmaps(Nmaps, i, ind, Bins, pixcoord, rad_dist, dist_err, percentile, weight, uu_ind, over_ind, OverError)
                
                uu_ind = iup
            # end apply error if test

            # call plotting
            Tools.PlotMaps(pixel,Ns, Npix, w, rad, Bins, i, filter)
            
            Nside_old = Ns
            
            ##################
        # end Nmaps part

    if Nmaps == None:
        """
        Use Bin_seq, no Nmaps as input.
        Bin_seq, Cutoff, rad_dist, dist_err, percentile, weight, ii, filter
        """
        print('Manually defined bins to map the stars, with bin edges:')
        print(Bin_seq)

        # test cut off
        if cutoff == None:
            cutoff = len(Bin_seq)
        if cutoff < 1:
            print('Cut off too low. Exit')
            sys.exit()
        if  cutoff > len(Bin_seq):
            print('Cut off too high. Exit')
            sys.exit()

        else:
            print('Use {} bins in distributing the stars after distance.'.format(cutoff-1))
        
        
        # Need the indexes going into each bin 
        # Cut off at different distances, able to plot at each distance. Do not save the data for each distance
        
        inte_ind       = np.where(percentile > 0.01)
        uu_ind, ll_ind = [], []
        over_ind       = np.where((percentile > 1) & (percentile < 10) & (rad_dist < 100*Bin_seq[-1]) & (rad_dist > Bin_seq[-1]))
        over_ind       = over_ind[0]
        OverUncer      = percentile[over_ind]
                
        # Loop over the the number of maps
        for i in range(len(Bin_seq)-1):

            print('==================')
            ind = np.where((rad_dist >= Bin_seq[i]) & (rad_dist < Bin_seq[i+1]))
            rad = rad_dist[ind]
            print('Bin number {}'.format(i))
            
            # Find an approproate Nside:
            Ns, Npix = Tools.Find_Nside(Nside, len(rad))
            
            # load pix coordinate for the given Nside
            if Ns == Nside_old:
                pass
            else:
                # read file
                print('read file for Nside {}'.format(Ns))
                f2       = h5py.File('SPC_Nside_{}.h5'.format(Ns),'r')
                pixcoord = np.asarray(f2['Healpix coordinates'])
                f2.close()
                pixcoord = pixcoord[ii]

            ###
            # call Apply error
            if filter == None:
                pass
            
            else:
                # Apply error:
                pix, w, rad, iup = Tools.ApplyErrorBinSeq(Bin_seq, i, ind, pixcoord, rad_dist, dist_err, percentile, weight, uu_ind, over_ind, cutoff, OverError)
                uu_ind = iup

            # end apply error testing
                    
            # Plotting
            Tools.PlotMaps(pix, Ns, Npix, w, rad, Bin_seq, i, filter)

            Nside_old = Ns
            if i == cutoff-1:
                print('Cut off reached')
                break        
        
    
    ######

    plt.show()

    # end DistributeDistance()
    
   
       

"""
Function calls
"""
# params:
Nside = 1024   # 128, 64 are test Nside?
Bin_seq = [1,100, 500, 1000,2000,3000,4000,4250,4500,4750,5000,6000]#,8000,10000,15000,20000,30000,50000,1e5,5e5,1e6]
print('----------------')

#############
start = time.time()

#----------------------#
# Make 3D map
B_outer = [3e4, 5e4,1e5,5e5,1e6,2e12]
#DistributeDistance(1024, Bin_seq = B_outer, filter='mean')
#DistributeDistance(1024, Nmaps=10, Rmax=1e3, Rmin=0.1, filter='mean')
#DistributeDistance(1024, Nmaps=30, Rmax=3e4, Rmin=0.1, filter='mean')
#DistributeDistance(1024, Nmaps=1, Rmax=1e4, Rmin=0.1, h=True)

#----------------------#
#Filters:
#for f in ['mean', 'BP', 'RP']:
#    AllStars(Nside, f)

#IntensityMapping(Nside, 'mean')
#IntensityMapping(Nside, 'G')
#IntensityMapping(Nside, 'BP')
#IntensityMapping(Nside, 'RP')


#----------------------#
# Distance distributions:
b = [1,50,100,250,500,750,1000,1500,2000,2500,3000,3500,4000,5000,6000,7000,8000,9000,10000,12500,15000,17500,20000,25000,30000,35000,40000,50000]#,75000,1e5,5e5,1e6,1e7,1e13]
#AllDistances(b, xscale='log', yscale='log')
#Tools.PlotDist_DistError()


#----------------------#
# Nside testing:
#testNside(10, 11)



############
end = time.time()
print('Run time: {}'.format(end-start))
print('---------------------')
