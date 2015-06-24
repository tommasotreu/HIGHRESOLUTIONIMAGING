"""
This file is distributed as part of the pylens/imageSim library under
the MIT License (http://opensource.org/licenses/MIT). Its use is
encouraged but not supported by the authors.

Copyright Matthew W. Auger and Xiao-Lei Meng, 2011, 2013, 2015

If you use this code in your research, please cite one or more of the
following papers:

Auger et al (2011) http://adsabs.harvard.edu/abs/2011MNRAS.411L...6A
Auger et al (2013) http://adsabs.harvard.edu/abs/2013MNRAS.436..503A
Meng et al (2015)  http://arxiv.org/abs/1506.XXXXX
"""
import numpy

def coords(shape):
    return numpy.indices(shape).astype(numpy.float64)

def overSample(shape,factor):
    coords = numpy.indices((shape[0]*factor,shape[1]*factor)).astype(numpy.float64)/factor - 0.5*(factor-1.)/factor
    return coords[0],coords[1]

def resamp(a,factor,add=False):
    arr = a.copy()
    x = arr.shape[1]/factor
    y = arr.shape[0]/factor
    o = numpy.zeros((y,x))
    for i in range(factor):
        for j in range(factor):
            o += arr[i::factor,j::factor]
    if add==True:
        return o
    return o/factor**2

def resampN(arr,factor,add=False):
    a = arr.copy()
    ndim = arr.ndim
    oshape = []
    for i in arr.shape:
        oshape.append(i/factor)
    o = numpy.zeros(oshape)
    cmd = ""
    addcmd = "    o += a["
    for i in range(ndim):
        ind = ' '*i
        cmd = '%s\n%sfor i%d in range(factor):'%(cmd,ind,i)
        addcmd = " %si%d::factor,"%(addcmd,i)
    cmd = '%s\n%s]'%(cmd,addcmd[:-1])
    exec cmd
    if add:
        return o
    return o/factor**ndim

resample = resamp

def recube(a,factor):
    arr = a.copy()
    x = arr.shape[1]/factor
    y = arr.shape[0]/factor
    o = numpy.empty((x*y,factor,factor))
    for i in range(factor):
        for j in range(factor):
            o[:,i,j] = arr[i::factor,j::factor].ravel()
    return o

