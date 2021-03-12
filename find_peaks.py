from numpy import diff, where, argmax, sign, mod, zeros, bitwise_and, transpose

def _find_peaks1d(data, method='zero_crossing'):
    if method =='zero_crossing':
        ind0 = where(diff(sign(data))!=0)[0]  
        N = len(ind0)-1
        extrema_indices = zeros(N, dtype=int)
        classify_intervals = zeros(N, dtype=int)
        for i in range(0, N):
            extrema_indices[i] = ind0[i] + argmax(data[ind0[i]:ind0[i+1]+1])        
            classify_intervals[i] = sum(data[ind0[i]+1:ind0[i+1]+1]<0)==0 # count number of negative values in the interval, the first may be negative though correct => excluded
        take = where(classify_intervals==1)[0]
        peak_indices = extrema_indices[take]
        
    elif method =='all_peaks':
        peak_indices = where(diff(sign(diff(data)))==-2)[0] + 1
    else:
        print('Your method name is not implemented, check spelling')
        return 0        
    # in bad data (measurements with jumps, things may go wrong (eg. for all_peaks) and the negative peaks are sorted away        
    #take = where(data[peak_indices]>0)
    #peak_indices = peak_indices[take]
    return peak_indices
        
        
def find_peaks(data, axis=0, method='zero_crossing'):
    '''
    Returns the indices where the data has peaks according to the given methods:
    The maximum is calculated in one dimension indicated by the axis
    So far limited to 2d, first dimension is the interesting one!
    Parameters:
    -----------
    input:
    -------
    data        array of input data 
    axis        int: defines axis for calculation of peak
    method      method for finding peaks: "zero_crossing": the peak between two zero crossings; zeros crossings does not find single peaks with negative values to each side
                method "all_peaks" finds all individiual peaks
    
    return:     1d array for 1d data input, two arrays for 2d data input
    --------
    peak_indices            
    '''
    
    if axis==1:
        data = data.transpose()    
    data_shape = data.shape  
    if len(data_shape)>2:
        print('Error: The method is not implemented for data.shape>2')
        return None
    elif len(data_shape)==2:
        data = transpose(data).flatten()
        N0 = data_shape[0] 
        peak_indices = _find_peaks1d(data, method)
        peak_indices_basic = mod(peak_indices, data_shape[0])
        peak_indices_second = (peak_indices/N0).astype('int')  
        take = where(bitwise_and(peak_indices_basic>0, peak_indices_basic<N0-1))[0]      # remove line jumps   
        if axis==0:
            return peak_indices_basic[take], peak_indices_second[take]
        else:
            return peak_indices_second[take], peak_indices_basic[take]
    else:
        N0 = len(data)   
        return _find_peaks1d(data, method)
        peak_indices_basic = mod(peak_indices, data_shape[0])
        peak_indices_second = (peak_indices/N0).astype('int')  
        take = where(bitwise_and(peak_indices_basic>0, peak_indices_basic<N0-1))[0]      # remove line jumps
        return peak_indices_basic[take], peak_indices_second[take]
        #return peak_indices_basic, peak_indices_second
    
    
    
if __name__=='__main__':
    from numpy import linspace, sin, pi
    #1d
    '''
    x = linspace(0,2*pi, 100)
    data = sin(3*x) + sin(5*x-0.3)
    peak_indices = find_peaks(data)
    peak_indices2 = find_peaks(data, method='all_peaks')
    p
    import pylab as plt
    plt.plot(x, data)
    plt.plot(x[peak_indices], data[peak_indices], '+')
    plt.plot(x[peak_indices2], data[peak_indices2], 'x')
    plt.show()
    '''
    #2d
    from numpy import meshgrid
    x = linspace(0,2*pi, 100)
    t = linspace(0,10, 100)
    X, T = meshgrid(x, t, indexing='ij')
    data = sin(3*X-2*T) + sin(5*X-10*T - 0.3)
    axis=1
    indx, indt = find_peaks(data, axis=axis)
    indx2, indt2 = find_peaks(data, method='all_peaks', axis=axis)
    import pylab as plt
    for i in range(0,100, 10):
        plt.figure()
        choose = i
        if axis==0:
            take = where(indt==choose)[0]
            take2 = where(indt2==choose)[0]
            plt.plot(x, data[:, choose])
            plt.plot(x[indx[take]], data[indx[take], choose], '+')
            #plt.plot(x[indx2[take2]], data[indx2[take2], choose], 'x')

        else:
            take = where(indx==choose)[0]
            take2 = where(indx2==choose)[0]
            plt.plot(t, data[choose,:])
            plt.plot(t[indt[take]], data[choose, indt[take]], '+')
            #plt.plot(t[indt2[take2]], data[choose, indt2[take2]], 'x')

    plt.show()
    #'''
        
