import numpy

def convolve(image,psf,doPSF=True,edgeCheck=False):
    """
    A reasonably fast convolution routine that supports re-entry with a
    pre-FFT'd PSF. Returns the convolved image and the FFT'd PSF.
    """
    datadim1 = image.shape[0]
    datadim2 = image.shape[1]
    if datadim1!=datadim2:
        ddim = max(datadim1,datadim2)
        s = numpy.binary_repr(ddim-1)
        s = s[:-1]+'0' # Guarantee that padding is used
    else:
        ddim = datadim1
        s = numpy.binary_repr(ddim-1)
    if s.find('0')>0:
        size = 2**len(s)
        if edgeCheck==True and size-ddim<8:
            size*=2
        boxd = numpy.zeros((size,size))
        r = size-datadim1
        r1 = r2 = r/2
        if r%2==1:
            r1 = r/2+1
        c = size-datadim2
        c1 = c2 = c/2
        if c%2==1:
            c1 = c/2+1
        boxdslice = (slice(r1,datadim1+r1),slice(c1,datadim2+c1))
        boxd[boxdslice] = image
    else:
        boxd = image

    if doPSF:
        # Pad the PSF to the image size
        boxp = boxd*0.
        if boxd.shape[0]==psf.shape[0]:
            boxp = psf.copy()
        else:
            r = boxp.shape[0]-psf.shape[0]
            r1 = r/2+1
            c = boxp.shape[1]-psf.shape[1]
            c1 = c/2+1
            boxpslice = (slice(r1,psf.shape[0]+r1),slice(c1,psf.shape[1]+c1))
            boxp[boxpslice] = psf.copy()
        # Store the transform of the image after the first iteration
        a = (numpy.fft.rfft2(boxp))
    else:
        a = psf
        # PSF transform and multiplication
    b = a*numpy.fft.rfft2(boxd)
    # Inverse transform, including phase-shift to put image back in center;
    #   this removes the requirement to do 2x zero-padding so makes things
    #   go a bit quicker.
    b = numpy.fft.fftshift(numpy.fft.irfft2(b)).real
    # If the image was padded, remove the padding
    if s.find('0')>0:
        b = b[boxdslice]

    return b,a


def prep(image,psf):
    datadim1 = image.shape[0]
    datadim2 = image.shape[1]
    if datadim1!=datadim2:
        ddim = max(datadim1,datadim2)
        s = numpy.binary_repr(ddim-1)
        s = s[:-1]+'0' # Guarantee that padding is used
    else:
        ddim = datadim1
        s = numpy.binary_repr(ddim-1)
    if s.find('0')>0:
        size = 2**len(s)
        boxd = numpy.zeros((size,size))
        r = size-datadim1
        r1 = r2 = r/2
        if r%2==1:
            r1 = r/2+1
        c = size-datadim2
        c1 = c2 = c/2
        if c%2==1:
            c1 = c/2+1
        boxdslice = (slice(r1,datadim1+r1),slice(c1,datadim2+c1))
        boxd[boxdslice] = image
    else:
        boxd = image

    boxp = boxd*0.
    if boxd.shape[0]==psf.shape[0]:
        boxp = psf.copy()
    else:
        r = boxp.shape[0]-psf.shape[0]
        r1 = r/2+1
        c = boxp.shape[1]-psf.shape[1]
        c1 = c/2+1
        boxpslice = (slice(r1,psf.shape[0]+r1),slice(c1,psf.shape[1]+c1))
        boxp[boxpslice] = psf.copy()

    from pyfft.cuda import Plan
    import pycuda.driver as cuda
    from pycuda.tools import make_default_context
    import pycuda.gpuarray as gpuarray
    cuda.init()
    context = make_default_context()
    stream = cuda.Stream()

    plan = Plan(boxp.shape,stream=stream)
    gdata = gpuarray.to_gpu(boxp.astype(numpy.complex64))
    plan.execute(gdata)
    return gdata,boxd.shape,boxdslice,plan,stream


def cConvolve(image,plan):
    from pyfft.cuda import Plan
    import pycuda.driver as cuda
    from pycuda.tools import make_default_context
    import pycuda.gpuarray as gpuarray

    if type(plan)==type(image):
        return prep(image,plan)

    psf,oshape,boxdslice,plan,stream = plan
    if image.shape!=oshape:
        im = numpy.zeros(oshape)
        im[boxdslice] = image
    else:
        im = image
    gdata = gpuarray.to_gpu(im.astype(numpy.complex64))
    plan.execute(gdata)
    o = gdata*psf
    plan.execute(o,inverse=True)
    o = numpy.fft.fftshift(o.get()).real
    if oshape!=image.shape:
        o = o[boxdslice]
    return o
