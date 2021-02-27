from numpy import argwhere, sqrt, var
from wave_tools import find_peaks

def find_freak_waves(eta, axis=0, method='crest_over_Hs', factor=1.25):
    '''
    Find 
    '''
    
    Hs = 4*sqrt(var(eta))   
    
    if len(eta.shape)==2:        
        crests0, crests1 = find_peaks.find_peaks(eta, axis)
        pick_freaks = argwhere(eta[crests0,crests1]>factor*Hs).transpose()[0]
        freak0, freak1 = crests0[pick_freaks], crests1[pick_freaks]
        return freak0, freak1, eta[freak0, freak1]
    elif len(eta.shape)==1:
        crests = find_peaks.find_peaks(eta, axis=0)
        pick_freaks = argwhere(eta[crests]>factor*Hs).transpose()[0]
        freak = crests[pick_freaks]
        return freak, eta[freak]
    else:
        print('The data shape that was used in the fuction is not of 1 or 2 dimensions. Higher dimensions have to be implemented!')       
