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

