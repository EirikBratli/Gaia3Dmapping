import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import sys, os, glob, time
import h5py
import Datahandling as dh
from matplotlib import cm
from pathlib import Path

import itertools as it

import Tools
import numba as nb
cmaps = cm.get_cmap

"""
Make distribution of stars within different distances. So can make a 3D map of
the distributions.



############
Functions to calculate properties
############
"""
# Params:
ZP_blue        = 25.3513881707
ZP_red         = 24.7619199882
ZP_green       = 25.6883657251




"""
#############
Functions to get wanted results:
#############
"""

def testNside(l,u):
    """
    Test different Nsides resolutions of the maps, see what is best. 
    Input: - l, int-scalar,
           - u, int-scalar,    Both must follow 2**p.
    """
    if l==False and u==False:
        print('No input! Need lower and upper power of 2, in Nside=2**p')
        sys.exit()
    elif l==False:
        print('No lower limit, type in first argument. Nside=2**p')
        sys.exit()
    elif u==False:
        print('No upper limit, type in second argument. Nside=2**p')
        sys.exit()
    else:
        pass
        

    if l<0 or u<0:
        print('Power must be positive')
        sys.exit()
    elif l>=12 or u>12:
        print('Power too high, max p=12. Will use too long time to generate maps')
    else:
        pass

    savepath = ''
    f1 = 'RightAscension.h5'
    f2 = 'Declination.h5'

    for p in range(l,u+1):
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
            print(len(pixpos))
        
        else:    
            print('Create coordinate file for Nside = {}'.format(Ns))
            pixpos = dh.PixelCoord(Ns, savepath+'RightAscension.h5', savepath+'Declination.h5', savepath)
            print(len(pixpos))

        Npix = hp.nside2npix(Ns)
        #map, bin_edges = np.histogram(pixpos, Npix)
        map, b = Map(pixpos, Npix)
        # plot
        hp.mollview(map, coord=['C','G'], nest=False, title='All stars with Nside={}'.format(Ns), unit='N_stars')
        plt.savefig('Figures/AllStars_diff_Nside{}.png'.format(Ns))

        t2 = time.time()
        print('Time used for Nside {} is: {}'.format(Ns, t2-t1))
        print('_______________')

    #####    

    plt.show()
    # end testNside


def AllStars(Nside, filter):
    """
    Plot all stars in one map. And make a distribution of the distance to the stars.
    Input: Nside
    """

    print('Plot all stars in map and make a distance distribution')

    f       = h5py.File('SPC_Nside_{}.h5'.format(Nside), 'r')
    pixcoord = np.asarray(f['Healpix coordinates'])
    f.close()

    # Map of all stars, No filters
    Npix = hp.nside2npix(Nside)
    print(Nside, Npix, 12*Nside**2)
 
    m    = Tools.Map(pixcoord, Npix)    
    hp.mollview(m, coord=['C','G'], nest=False, title='All stars, Nside = {}'.format(Nside), unit='Nstars')
    plt.savefig('Figures/AllStars_Nside{}.png'.format(Nside))

    #"""
    # Add intensity. With filters, Green, Blue, Red and their mean:
    print('Include magnitude as weights, with green, blue, red and mean')

    # Find a way to add intensity
    w = Tools.AddIntensity(filter)
    
    MapG, bG = np.histogram(pixcoord, Npix, weights=w)
    print('Map Green')
    
    hp.mollview(np.log(MapG), coord=['C','G'], nest=False, title='All stars in green filter, Nside = {}'.format(Nside), unit='Nstars')
    plt.savefig('Figures/AllStars_{}_Nside{}.png'.format(filter, Nside))
   
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
    print(min(rad_dist), max(rad_dist))
    print(xscale, yscale)
    print(Bins)
    
    # Plot histogram of the distancec distribution
    plt.hist(rad_dist, bins=Bins)
    plt.gca().set_xscale(xscale)
    plt.gca().set_yscale(yscale)
    plt.xlabel('pc')
    plt.ylabel('Number of stars')
    plt.savefig('Figures/Distance_distribution_all.png')

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
    
    f1 = h5py.File('SPC_Nside_{}.h5'.format(Nside), 'r')
    pixcoord = np.asarray(f1['Healpix coordinates'])
    f1.close()

    if filter == None:
        pass
    
    if filter == 'mean':
        f2 = h5py.File('Mean_Mag_G.h5','r')
        Mag = np.asarray(f2['mean_mag_G'])
        f2.close()
        
        n          = np.isnan(Mag)        
        ifnan      = np.where(n==True) 
        Mag[ifnan] = ZP_green
        
    else:
        f2 = h5py.File('Mean_Mag_{}.h5'.format(filter),'r')
        Mag = np.asarray(f2['mean_mag_{}'.format(filter)])
        f2.close()

        n          = np.isnan(Mag)        
        ifnan      = np.where(n==True) 
        if filter == 'BP':
            Mag[ifnan] = ZP_blue
        if filter == 'RP':
            Mag[ifnan] = ZP_red
        if filter == 'G':
            Mag[ifnan] = ZP_green
            
        

    #f3 = h5py.File('Distance.h5','r')
    #dist = np.asarray(f3['distance'])
    #f3.close()
    #dist = np.nan_to_num(dist)
    #rad = np.abs(dist)
    
    w = Tools.AddIntensity(filter)

    Npix = hp.nside2npix(Nside)
    N = len(Mag)
    print(N, Npix)

    Bins = [26,20,15,10,5,0.0001]
    ind0 = np.where((Mag <= min(Bins))) # Check for stars with magnitude stronger than 0
    print(len(ind0[0]), np.min(Mag))        

    for i in range(len(Bins)-1):
        print('Magnitude range: {} to {}'.format(Bins[i], Bins[i+1]))

        ind = np.where((Mag < Bins[i]) & (Mag >= Bins[i+1]))

        m, b = np.histogram(pixcoord[ind], Npix)        
        hp.mollview(np.log(m), coord=['C','G'], nest=False, title='Magnitude form {} mag to {} mag'.format(Bins[i], Bins[i+1]), unit='Nstars')
        plt.savefig('Figures/test{}_Mag_dist_Nside{}.png'.format(i,Nside))

        Map, b = np.histogram(pixcoord[ind], Npix, weights=w[ind])
        hp.mollview(np.log(Map), coord=['C','G'], nest=False, title='Weighted for mag {} to {}'.format(Bins[i], Bins[i+1]), unit='Nstars')
        plt.savefig('Figures/test{}_wMag_{}_Nside{}.png'.format(i, filter, Nside))

        #plt.figure('hist {}'.format(i-1))
        #plt.hist(rad[ind], 100)
        #plt.xlabel('pc')
        #plt.gca().set_xscale('log')
        #plt.gca().set_yscale('log')
    
    #plt.show()
    # End IntensityMapping

    
    



def DistributeDistance(Nside, Nmaps=None, Rmax=5e4, Rmin=0.1, Bin_seq=None, cutoff=None, filter=None, h=False):
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

    #f2       = h5py.File('SPC_Nside_{}.h5'.format(Nside),'r')
    #pixcoord = np.asarray(f2['Healpix coordinates'])
    #f2.close()

    dist_err = Tools.DistError(dist)
    
    #print(dist_err)
    
    # check for nan values in dist:
    ii          = Tools.NoDistance(dist)
    dist        = dist[ii]
    dist_err    = dist_err[ii]
    print(len(dist), len(dist_err))
    
    
    # radial distance
    rad_dist = np.abs(dist)
    print(rad_dist)
    print(dist_err)
    Nside_old = 0
    
    # Distance percentile:
    percentile = dist_err/rad_dist
    print((percentile))

    over_ind = np.where((percentile > 1) & (percentile < 10))
    over_ind = over_ind[0]
    print(len(over_ind))
    OverUncer = percentile[over_ind]
    print(np.min(rad_dist[over_ind]))
    #sys.exit()

    
    # Add intesity:
    print('Load weights')
    weight = Tools.AddIntensity(filter)
    
    
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
        """
        print('Use {} numbers of maps to map the star distribution'.format(Nmaps))
        size = (Rmax-Rmin)/Nmaps
        Bins = np.arange(Rmin, Rmax+10, size)
        print(size, Rmin, Rmax, len(Bins))
        print(Bins)

        inte_ind = np.where(percentile > 0.01)
        uu_ind   = []
        print(len(inte_ind[0]))
        print(np.min(rad_dist[inte_ind]))

        """
        t1 = time.time()
        w0 = []
        p  = []
        r  = []
        for k in over_ind:
            #print(rad_dist[k], k)
            if (rad_dist[k] < 100*Bins[-1]) and rad_dist[k] > Bins[-1]:
                a         = Tools.IntegrateOvererror(rad_dist[k], dist_err[k], Bins[-1]) 
                weight[k] = a*weight[k]
                print(k, a, rad_dist[k], dist_err[k], percentile[k])
                r.append(rad_dist[k])
                w0.append(weight[k])
                #p.append(pixcoord[k])
            #end if
        #end for loop
        print(len(w0))
        t2 = time.time()
        print('time: {} s'.format(t2-t1))
        sys.exit()
        """
        for i in range(Nmaps):
            
            ind = np.where((rad_dist > Bins[i]) & (rad_dist <= Bins[i+1]))
            print('Bin number {}'.format(i), len(ind[0]))
            rad = rad_dist[ind]

            #"""
            # Find appropriate Nside
            if isinstance(Nside, list) == True:
                # Let the number of stars in a bin give the Nside
                #print('Nside is list of length', len(Nside))
                Nside.sort()
                #print(Nside)
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
                #print('Nside is scalar')
                Ns = Nside
            
            Npix = hp.nside2npix(Ns)
            
            # load pix coordinate for the given Nside
            if Ns == Nside_old:
                pass
            else:
                # read file
                print('read file for Nside {}'.format(Ns))
                f2       = h5py.File('SPC_Nside_{}.h5'.format(Ns),'r')
                pixcoord = np.asarray(f2['Healpix coordinates'])
                f2.close()

            pix = pixcoord[ind]
               
            #"""
            ######
            #174290352 j=368843250 1914.536 
            #345848636 j=731761252 1972.9777 
            
            if weight.all() == None:
                pass
            
            else:
                # Apply error:
                wu, wuu, wl, aa = 0,0,0,0
                l_ind, u_ind    = [],[]
                eps             = Bins[i+1]*0.01
                
                #ind = np.where((rad_dist <= Bins[i+1]) & (rad_dist > Bins[i]))
                pixel = pixcoord[ind]
                #r_err = dist_err[ind]
                per   = percentile[ind]
                w     = weight[ind]
                c     = 0
                t1 = time.time()
                #ii = np.where((percentile > 0.01) & (percentile < 1))
                iup = np.where((percentile > 0.01) & (percentile < 1) & (rad_dist > Bins[i+1]-eps) & (rad_dist <= Bins[i+1]))
                ilo = np.where((percentile > 0.01) & (percentile < 1) & (rad_dist <= Bins[i] +eps) & (rad_dist > Bins[i]))
                #iiupeq = np.where(iup==ind)
                print(iup[0], ilo[0])
                print('-->', i, len(iup[0]), len(ilo[0]))
                print(Bins[i]+eps, Bins[i+1]-eps)
                print(len(pixel), len(w))
                #sys.exit()
                
                for k,j in enumerate(ilo[0]):
                    #if percentile[j] > 0.01:
                    print(k,j,rad_dist[j], rad[k]) 
                    #    if rad_dist[j] < Bins[i] + eps:
                    #        #l_ind.append(j)
                            
                    #        # Integrate error
                    a         = Tools.IntegrateError(rad_dist[j], dist_err[j], Bins[i])
                    weight[j] = weight[j]*(1-a)
                    c += 1
                    print('lower:', k, percentile[j], a)
                    # end if
                    # end if
                # end lower bin loop
                for k,j in enumerate(iup[0]):
                    print(k,j, rad_dist[j], rad[k])
                    #if percentile[j] > 0.01:
                    #    if rad_dist[j] > Bins[i+1] - eps:
                    #u_ind.append(j)
                    c += 1
                    # integrate error
                    a         = Tools.IntegrateError(rad_dist[j], dist_err[j], Bins[i+1])
                    weight[j] = weight[j]*a
                    wuu       = weight[uu_ind]*(1-a)
                    print('upper:', k, percentile[j], a)
                    # end if
                    # end if
                ## end upper bin loop
                t2 = time.time()
                print('k loop time: {} s'.format(t2-t1), c)
                print(len(l_ind), len(u_ind), len(uu_ind))
                if len(uu_ind) != 0:
                    pixel = np.concatenate((pixel, pixcoord[uu_ind]), axis=0)
                    w     = np.concatenate((weight[ind], wuu), axis=0)
                    rad   = np.concatenate((rad, rad_dist[uu_ind]), axis=0)

                else:
                    pixel = pixel
                    w     = weight[ind]
                    rad   = rad
                #end if
                print(len(pixel), len(w))
                #sys.exit()
                """
                
                for k,j in enumerate(ind[0]):
                
                    if percentile[j] > 0.01:
                        print(k, j, rad_dist[j], rad[k])
                        if rad_dist[j] < Bins[i] + eps:
                            #l_ind.append(j)
                            
                            # Integrate error
                            a         = Tools.IntegrateError(rad_dist[j], dist_err[j], Bins[i])
                            weight[j] = weight[j]*(1-a)
                            c += 1
                            print('lower:', k, percentile[j], a)
                        elif rad_dist[j] > Bins[i+1] - eps:
                            #u_ind.append(j)
                            c += 1
                            # integrate error
                            a         = Tools.IntegrateError(rad_dist[j], dist_err[j], Bins[i+1])
                            weight[j] = weight[j]*a
                            wuu       = weight[uu_ind]*(1-a)
                            print('upper:', k, percentile[j], a)
                            # end if
                    # end if
                ## end k loop
                t2 = time.time()
                print('k loop time: {} s'.format(t2-t1), c)
                print(len(l_ind), len(u_ind), len(uu_ind))
                if len(uu_ind) != 0:
                    pixel = np.concatenate((pixel, pixcoord[uu_ind]), axis=0)
                    w     = np.concatenate((weight[ind], wuu), axis=0)
                    rad   = np.concatenate((rad, rad_dist[uu_ind]), axis=0)

                else:
                    pixel = pixel
                    w     = weight[ind]
                    rad   = rad
                #end if
                print(len(pixel), len(w))
                ##
                #"""
                """
                if i == Nmaps-1:
                    w0 = []
                    p  = []
                    r  = []
                    for k in over_ind:
                        #print(rad_dist[k], k)
                        if (rad_dist[k] < 100*Bins[-1]) and rad_dist[k] > Bins[-1]:
                            a         = Tools.IntegrateOvererror(rad_dist[k], dist_err[k], Bins[-1]) # why slow??
                            weight[k] = a*weight[k]
                            #print(k, a)
                            r.append(rad_dist[k])
                            w0.append(weight[k])
                            p.append(pixcoord[k])
                        #end if
                    #end for loop
                    pixel = np.concatenate((pixel, p), axis=0)
                    w     = np.concatenate((w, w0), axis=0)
                    rad   = np.concatenate((rad, r), axis=0)
                    print(len(pixel), len(weight))
                    
                #"""
                #uu_w   = u_weight
                uu_ind = u_ind

            # end apply error if test
            #######
            
            #"""
            print(len(pix), len(w))
            if filter == None:
                logmap    = Tools.Map(pix, Npix)

            else:
                map, b = np.histogram(pix, Npix, weights=w)
            
                logmap = np.log(map)
                logmap[np.isinf(logmap)] = -60
                
            hp.mollview(logmap, coord=['C','G'], nest=False, title='Stars from {} pc to {} pc, Nside {}'.format(round(Bins[i]), round(Bins[i+1]), Ns), unit='Nstars')
            
            #plt.savefig('Figures/Maplayer{}pc_size{}_{}_Nside{}.png'.format(int(round(Bins[i+1])), int(round(size)), filter, Ns))
            #hp.cartview(logmap, coord=['C','G'], title='cart: {} pc to {} pc'.format(round(Bins[i]),round(Bins[i+1])))
            if h == True:
                plt.figure('Histogram {}'.format(i))
                plt.hist(rad, bins=50)
                plt.xlabel('pc', size=14)
                plt.ylabel('Number of stars', size=14)
                plt.savefig('Figures/Hist_maplayer{}_Nside{}.png'.format(i,Ns))
            
            ##
            #"""
            Nside_old = Ns
            
            ##################

    if Nmaps == None:
        """
        Use Bin_seq, no Nmaps as input.
        """
        print('Manually defined bins to map the stars')
        print(Bin_seq)

        # test bin_seq:
        if max(Bin_seq) > np.max(rad_dist):
            Bin_seq = Bin_seq
        else:
            Bin_seq.append(np.max(rad_dist)+1)
            print('Add max limit to bin sequence, length of bin list is {}'.format(len(Bin_seq)))

        # test cut off
        if cutoff == None:
            cutoff = len(Bin_seq[:20])
        if cutoff < 1:
            print('Cut off too low. Exit')
            sys.exit()
        if  cutoff > len(Bin_seq):
            print('Cut off too high. Exit')
            sys.exit()

        else:
            print('Use {} bins in distributing the stars after distance.'.format(cutoff))
        
        
        # Need the indexes going into each bin 
        # Cut off at different distances, able to plot at each distance. Do not save the data for each distance
        
        for i in range(len(Bin_seq)-1):
            
            ind = np.where((rad_dist >= Bin_seq[i]) & (rad_dist < Bin_seq[i+1]))
            rad = rad_dist[ind]
            print('Bin number {}'.format(i))

            if isinstance(Nside, list) == True:
                # Let the number of stars in a bin give the Nside
                #print('Nside is list of length', len(Nside))
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
                #print('Nside is scalar')
                Ns = Nside
               
            #"""
            print(len(rad))
            Npix = hp.nside2npix(Ns)
            
            # load pix coordinate for the given Nside
            if Ns == Nside_old:
                pass
            else:
                # read file
                print('read file for Nside {}'.format(Ns))
                f2       = h5py.File('SPC_Nside_{}.h5'.format(Ns),'r')
                pixcoord = np.asarray(f2['Healpix coordinates'])
                f2.close()


            ###
            pix = pixcoord[ind]
            #if i==0:
            #    pos_map, b = np.histogram(pix, Npix)
            #    hp.mollview(pos_map, nest=False, title='Stars with unknown distance', unit='Nstars')
            #else:
            if Bin_seq[i] >= 1:
                if filter == None:
                    pos_map, b = Tools.Map(pix,Npix)
                else:
                    pos_map, b = np.histogram(pix, Npix, weights=w[ind])

                logmap = np.log(map)
                logmap[np.isinf(logmap)] = -60

                hp.mollview(np.log(pos_map), coord=['C','G'],nest=False, title='Stars form {} pc to {} pc, Nside={}'.format(Bin_seq[i], Bin_seq[i+1], Ns), unit='Nstars')
                #plt.savefig('Figures/Irr_Maplayer{}_Nside{}.png'.format(i,Ns))
                if h == True:
                    plt.figure('Histogram {}'.format(i))
                    plt.hist(rad, bins=50)
                    plt.xlabel('pc')
                    plt.ylabel('Number of stars')
                    plt.savefig('Figures/Irr_MaplayerHist{}_Nside{}.png'.format(i,Ns))

            Nside_old = Ns
            if i == cutoff-1:
                print('Cut off reached')
                break        
        
    
    ######
    plt.show()

    
   
       

"""
Function calls
"""
# params:
Nside = 512   # 128, 64 are test Nside?
Bin_seq = [0, 1,100, 200,300, 400, 500, 750,1000,1250,1500,2000,2500,3000,4000,5000,6000,8000,10000,15000,20000,30000,50000,1e5,5e5,1e6]

#############
start = time.time()

#DistributeDistance([128, 256], Bin_seq = Bin_seq, cutoff=6)
DistributeDistance(512, Nmaps=5, Rmax=1e4, Rmin=0.1, filter='G')
#DistributeDistance(512, Nmaps=10, Rmax=1e4, Rmin=0.1) 

#AllStars(Nside)

#AllDistances(Bin_seq, xscale='log', yscale='linear')

#testNside(11,12)

#IntensityMapping(Nside, 'mean')
#IntensityMapping(Nside, 'G')
#IntensityMapping(Nside, 'BP')
#IntensityMapping(Nside, 'RP')


############
end = time.time()
print('Run time: {}'.format(end-start))

plt.show()
