#-------------------------------------------------------------------------------------------------------------
# IMPORTING MODULES
import numpy
import pylab
import sys
#sys.path.append("/home/xlmeng/python_scorza2/")
import convolve,SBModels
import pylens,MassModels
import indexTricks as iT
import pyfits
import pymc
from math import log10
from SampleOpt import AMAOpt
#------------ the following ones are for "getModel"
from scipy import optimize

#-------------------------------------------------------------------------------------------------------------
def wrapper(sys_para,instr_para,psfile_name):
    """
    To modulize pylens main function
    """
    zp_1s = instr_para[2]       # zero point for 1s
    ep = instr_para[7]          # exposre
    pix_num = instr_para[5]     # pixel number
    read_val = instr_para[3]    # readout noise 1s
    bg_val = instr_para[4]      # background noise 1s
    max_exp = instr_para[6]     # max exposure

# Create surface brightness objects that define the model
    lensGal = SBModels.Sersic('lens',{'x':sys_para[1],'y':sys_para[2],'re':sys_para[3],'q':sys_para[4],'pa':sys_para[5],'n':sys_para[6]})
    mag_lens = sys_para[7]
    zp = zp_1s+2.5*log10(ep)
    lensMag = lensGal.Mag(zp)
    lensGal.amp = 10.**(-0.4*(mag_lens-lensMag))

    src = SBModels.Sersic('src',{'x':sys_para[8],'y':sys_para[9],'re':sys_para[10],'q':sys_para[11],'pa':sys_para[12],'n':sys_para[13]})
    mag_src = sys_para[14]
    srcMag = src.Mag(zp)
    src.amp = 10.**(-0.4*(mag_src-srcMag))

# Create mass (or lensing) objects
    lensMass = MassModels.PowerLaw('lens',{'x':sys_para[15],'y':sys_para[16],'b':sys_para[17],'q':sys_para[18],'pa':sys_para[19],'eta':sys_para[20]})

    shear = MassModels.ExtShear('shear',{'x':sys_para[21],'y':sys_para[22],'b':sys_para[23],'pa':sys_para[24]})

    lenses = [lensMass,shear]

#create the coordinate grids that the image will be evaluated on
    y,x = iT.coords((pix_num,pix_num))

# create a PSF   
    psf = pyfits.open(psfile_name)[0].data
    psf /= psf.sum()

# add AGN
    agn1 = SBModels.PointSource('agn1',psf,{'x':sys_para[25],'y':sys_para[26]})
    mag_AGN = sys_para[31]
    zp = zp_1s+2.5*log10(ep)
    AGNMag = agn1.Mag(zp)
    agn1.amp = 10.**(-0.4*(mag_AGN-AGNMag))*sys_para[27]

    agn2 = SBModels.PointSource('agn2',psf,{'x':sys_para[28],'y':sys_para[29]})
    mag_AGN = sys_para[31]
    zp = zp_1s+2.5*log10(ep)
    AGNMag = agn2.Mag(zp)
    agn2.amp = 10.**(-0.4*(mag_AGN-AGNMag))*sys_para[30]

# Form the image and convolve (convolution returns the image and FFT'd PSF)
    img,psf = convolve.convolve(lensGal.pixeval(x,y),psf)

    img +=convolve.convolve(agn1.pixeval(x,y),psf,False)[0]
    img +=convolve.convolve(agn2.pixeval(x,y),psf,False)[0]

    xl,yl = pylens.getDeflections(lenses,[x,y])
    img +=convolve.convolve(src.pixeval(xl,yl),psf,False)[0]

# pyfits.PrimaryHDU(img).writeto('{0}.fits'.format(i+1),clobber=True)
    counts = numpy.array(img)

#add noise   
    read = numpy.zeros((pix_num,pix_num))
    read[:,:] = read_val*read_val*(int(ep/max_exp)+1.)
    read_array = numpy.array(read)

    bg = numpy.zeros((pix_num,pix_num))
    bg[:,:] = bg_val*ep
    bg_array = numpy.array(bg)

    variance = numpy.add(numpy.add(counts,read_array),bg_array)
    noise = numpy.power(variance,0.5)

#standard normal distribution
    normal = numpy.random.randn(pix_num,pix_num)
    normal_array = numpy.array(normal)
    noise_random = noise*normal_array
    img +=noise_random

# Now if you wanted to _fit_ an image you'd do mostly the same thing, but
#  use variables for the parameters instead of constants

# Define the variables for the lens galaxy light
    GX = pymc.Uniform('gx',sys_para[1]-sys_para[1]/10.,sys_para[1]+sys_para[1]/10.,value=sys_para[1])
    GY = pymc.Uniform('gy',sys_para[2]-sys_para[2]/10.,sys_para[2]+sys_para[2]/10.,value=sys_para[2])
    GR = pymc.Uniform('gr',1.,50.,value=sys_para[3])
    GQ = pymc.Uniform('gq',0.1,1.,value=sys_para[4])
    GP = pymc.Uniform('gp',-180.,180.,value=sys_para[5])
    GN = pymc.Uniform('gn',0.5,8.,value=sys_para[6])

# Variables for the source light; we'll start with a spherical n=1 object
    SX = pymc.Uniform('sx',sys_para[8]-sys_para[8]/10.,sys_para[8]+sys_para[8]/10.,value=sys_para[8])
    SY = pymc.Uniform('sy',sys_para[9]-sys_para[9]/10.,sys_para[9]+sys_para[9]/10.,value=sys_para[9])
    SR = pymc.Uniform('sr',1.,50.,value=sys_para[10])
    SQ = pymc.Uniform('sq',0.1,1.,value=sys_para[11])
    SP = pymc.Uniform('sp',-180.,180.,value=sys_para[12])
    SN = pymc.Uniform('sn',0.5,8.,value=sys_para[13])

# Variables for the mass models; we'll have a prior tying the mass position
#   to the light but we'll leave the ellipticity free here
    LX = pymc.Normal('lx',GX,1./3**2,value=GX.value) # Gaussian with sigma = 3
    LY = pymc.Normal('ly',GY,1./3**2,value=GY.value) # Use precision, not sigma...
    LB = pymc.Uniform('lb',1.,50.,value=sys_para[17])
    LQ = pymc.Uniform('lq',0.1,1.,value=sys_para[18])
    LP = pymc.Uniform('lp',-180.,180.,value=sys_para[19])
    LE = pymc.Uniform('le',0.7,1.3,value=sys_para[20])

    XB = pymc.Uniform('shear',0.,0.5,value=sys_para[23])
    XP = pymc.Uniform('shear pa',-180.,180.,value=sys_para[24])

# Now create the models as before, but with variables instead of constants!
    lensGal = SBModels.Sersic('lens',{'x':GX,'y':GY,'re':GR,'q':GQ,'pa':GP,'n':GN})
    src = SBModels.Sersic('src',{'x':SX,'y':SY,'re':SR,'q':SQ,'pa':SP,'n':SN})

    lens = MassModels.PowerLaw('lens',{'x':LX,'y':LY,'b':LB,'q':LQ,'pa':LP,'eta':LE})
# The shear origin will be on the mass
    shear = MassModels.ExtShear('shear',{'x':LX,'y':LY,'b':XB,'pa':XP})
    lenses = [lens,shear]

# Create a list for the parameters, and define starting proposal steps for
#   the optimization
    pars = [GX,GY,GR,GQ,GP,GN]
    cov = [0.05,0.05,0.1,0.01,3.,0.1]
    pars = pars+[SX,SY,SR,SQ,SP,SN]
    cov = cov+[0.1,0.1,0.1,0.01,3.,0.1]
    pars = pars+[LX,LY,LB,LQ,LP,LE,XB,XP]
    cov = cov+[0.05,0.05,0.05,0.01,3.,0.01,0.005,3.]

# This evaluates the model at the given parameters and finds the best
#   amplitudes for the surface brightness components

# Finally define the optimization function; this evaluates the model and the
#   current parameter values and returns the residual
    @pymc.observed
    def logP(value=0.,tmpXXX=GX): # This is a non-standard use of pymc....
        chi = getModel(img, noise, lensGal, src, lenses, x, y, psf)[0]
        return -0.5*chi**2

# Now optimize
    for i in range(2):
        sampler = AMAOpt(pars,[logP],[],cov=numpy.array(cov))
        sampler.sample(1500)
    logp,trace,result = sampler.result()

    for i in range(len(pars)):
        pars[i].value = trace[-1][i]

    return (result['lb'][-1], result['le'][-1])

#-------------------------------------------------------------------------------------------------------------
def getModel(img, noise, lensGal, src, lenses, x, y, psf):
    """
    cut from the original built-in subroutine at pylens main function, located after "S = noise.flatten()"
    original notes:
# We need to update the parameters of the objects to the values proposed
#   by the optimizer
    """
    I = img.flatten()
    S = noise.flatten()

    lensGal.setPars()
    src.setPars()
    for l in lenses:
        l.setPars()

    xl,yl = pylens.getDeflections(lenses,[x,y])

    model = numpy.empty((img.size,2))
    model[:,0] = convolve.convolve(lensGal.pixeval(x,y),psf,False)[0].ravel()/S
    model[:,1] = convolve.convolve(src.pixeval(xl,yl),psf,False)[0].ravel()/S

    amps,chi = optimize.nnls(model,I/S)
    return chi,amps*(model.T*S).T

#-------------------------------------------------------------------------------------------------------------
def run(sysfile_name,instr_para,psfile_name):
    """
    To conduct simulation and MCMC fitting iteratively
    """
    sys_para=numpy.genfromtxt(sysfile_name, dtype=float, comments='#')
    Nobj=len(sys_para)
    for i in xrange(Nobj):
        print '#------------------------ begin run No. %s' % str(i+1)
        (lb_lastone,le_lastone)=wrapper(sys_para[i,:],instr_para,psfile_name)
        print 'lb_lastone=%s,  le_lastone=%s' % (str(lb_lastone), str(le_lastone))
        print 'simulation and MCMC fitting No. '+str(i+1)+' completed!'

#
##        print 'No. '+str(i+1)+' simulation done!'
##        pylab.imshow(img,origin='lower',interpolation='nearest')
##        pylab.colorbar()
##        pylab.show()
##        pylab.ion()
