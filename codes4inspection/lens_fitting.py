#======================================================================================================================
# IMPORTING MODULES
import numpy
import pylab
import pyfits
import pymc
import sys
sys.path.append("/home/xlmeng/python_scorza2/")
import convolve,SBModels
import pylens,MassModels
import indexTricks as iT
import cPickle
from math import log10
#------------ the following ones are for sampler and optimizer
from SampleOpt import AMAOpt,Sampler
#------------ the following ones are for "getModel"
from scipy import optimize

#======================================================================================================================
def wrapper(sys_para, instr_para, ep_para, psf_name, Nsample=1500, flag=True):
    """
    This subroutine modulizes the main function "example4xiaolei.py" of the pylens software package
    
    --- INPUT ---
    sys_para    :   an ndim=1 array containing parameters for the lens+source systems
    instr_para  :   an ndim=1 array containing parameters for the observational instruments (HST, LSST, etc.)
    ep_para     :   an ndim=1 array containing parameters for the exposures
    psf_name    :   a string leading to the path+name of the psf fits file
    Nsample     :   an int number giving the sample size of MCMC fitting
    flag        :   a boolean deciding whether to run MCMC ('True') ot not ('False', in this case showing simulated images instead)

    === OUTPUT ===
    rslt = sampler.result()     :   an n=3 tuple, if flag=False, it is (None,None,None), otherwise
                            [0] - logP   : ndarray, shape=(Nsample,) , recording the log(posterior)
                            [1] - trace  : ndarray, shape=(Nsample, num of param in MCMC), recording param values
                            [2] - result : dict type quantity, basically the same with [1], but with header info included
    """
    zp_1s = instr_para[2]       # zero point for 1s
    ep = ep_para[0]             # exposre
    pix_num = instr_para[5]     # pixel number
    read_val = instr_para[3]    # readout noise 1s
    bg_val = instr_para[4]      # background noise 1s
    max_exp = instr_para[6]     # max exposure

    #--------------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------------
    # Create mass (or lensing) objects
    lensMass = MassModels.PowerLaw('lens',{'x':sys_para[15],'y':sys_para[16],'b':sys_para[17],'q':sys_para[18],'pa':sys_para[19],'eta':sys_para[20]})
    shear = MassModels.ExtShear('shear',{'x':sys_para[21],'y':sys_para[22],'b':sys_para[23],'pa':sys_para[24]})
    lenses = [lensMass,shear]

    #--------------------------------------------------------------------------------------
    #create the coordinate grids that the image will be evaluated on
    y,x = iT.coords((pix_num,pix_num))

    #--------------------------------------------------------------------------------------
    # create a PSF   
    psf = pyfits.open(psf_name)[0].data
    psf /= psf.sum()

    #--------------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------------    
    # Form the image and convolve (convolution returns the image and FFT'd PSF)
    img,psf = convolve.convolve(lensGal.pixeval(x,y),psf)

    #--------------------------------------------------------------------------------------
    # add AGNs to the image
#    img +=convolve.convolve(agn1.pixeval(x,y),psf,False)[0]
#    img +=convolve.convolve(agn2.pixeval(x,y),psf,False)[0]
    img +=agn1.pixeval(x,y)
    img +=agn2.pixeval(x,y)

    xl,yl = pylens.getDeflections(lenses,[x,y])
    img +=convolve.convolve(src.pixeval(xl,yl),psf,False)[0]

# pyfits.PrimaryHDU(img).writeto('{0}.fits'.format(i+1),clobber=True)
    counts = numpy.array(img)

    #--------------------------------------------------------------------------------------    
    #add noise
    read = numpy.zeros((pix_num,pix_num))
    read[:,:] = read_val*read_val*(int(ep/max_exp)+1.)
    read_array = numpy.array(read)

    bg = numpy.zeros((pix_num,pix_num))
    bg[:,:] = bg_val*ep
    bg_array = numpy.array(bg)

    variance = numpy.add(numpy.add(counts,read_array),bg_array)
    noise = numpy.power(variance,0.5)

    #--------------------------------------------------------------------------------------
    #standard normal distribution
    normal = numpy.random.randn(pix_num,pix_num)
    normal_array = numpy.array(normal)
    noise_random = noise*normal_array
    img +=noise_random

    #--------------------------------------------------------------------------------------    
    # a switch to turn on (flag=True) MCMC or not
    if flag == False:
        # Plotting img
        pylab.imshow(img,origin='lower',interpolation='nearest')
        pylab.colorbar()
        # pylab.ion()
        pylab.show()
        rslt = (None,None,None)

    else:
    #--------------------------------------------------------------------------------------
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

        # Variables for AGN's positions
        AX1 = pymc.Uniform('ax1',sys_para[25]-sys_para[25]/10.,sys_para[25]+sys_para[25]/10.,value=sys_para[25])
        AY1 = pymc.Uniform('ay1',sys_para[26]-sys_para[26]/10.,sys_para[26]+sys_para[26]/10.,value=sys_para[26])
        AX2 = pymc.Uniform('ax2',sys_para[28]-sys_para[28]/10.,sys_para[28]+sys_para[28]/10.,value=sys_para[28])
        AY2 = pymc.Uniform('ay2',sys_para[29]-sys_para[29]/10.,sys_para[29]+sys_para[29]/10.,value=sys_para[29])

        # Now create the models as before, but with variables instead of constants!
        lensGal = SBModels.Sersic('lens',{'x':GX,'y':GY,'re':GR,'q':GQ,'pa':GP,'n':GN})
        src = SBModels.Sersic('src',{'x':SX,'y':SY,'re':SR,'q':SQ,'pa':SP,'n':SN})

        lens = MassModels.PowerLaw('lens',{'x':LX,'y':LY,'b':LB,'q':LQ,'pa':LP,'eta':LE})
        # The shear origin will be on the mass
        shear = MassModels.ExtShear('shear',{'x':LX,'y':LY,'b':XB,'pa':XP})
        lenses = [lens,shear]

        # The ANG
        agn1 = SBModels.PointSource('agn1',psf,{'x':AX1,'y':AY1})
        agn2 = SBModels.PointSource('agn2',psf,{'x':AX2,'y':AY2})

        # Create a list for the parameters, and define starting proposal steps for the optimization
        pars = [GX,GY,GR,GQ,GP,GN]
        cov = [0.05,0.05,0.1,0.01,3.,0.1]
        pars = pars+[SX,SY,SR,SQ,SP,SN]
        cov = cov+[0.1,0.1,0.1,0.01,3.,0.1]
        pars = pars+[LX,LY,LB,LQ,LP,LE,XB,XP]
        cov = cov+[0.05,0.05,0.05,0.01,3.,0.01,0.005,3.]
        pars = pars+[AX1,AY1,AX2,AY2]
        cov = cov+[0.05,0.05,0.05,0.05]

        # Finally define the optimization function; this evaluates the model and the
        #   current parameter values and returns the residual
        print 'marker before pymc.observed'
        @pymc.observed
        def logP(value=0.,tmpXXX=GX): # This is a non-standard use of pymc....
##################### QUESTION 1: is the passing of "tmpXXX=GX" ok for my MCMC sampling of parameters LB and LE?
#        def logP(value=0.,sampleparam=[LB,LE]):
            chi = getModel(img, noise, lensGal, src, lenses, x, y, agn1, agn2, psf)[0]
            print '-2ln(likelihood) = chi^2 = %s' % str(chi**2)
            return -0.5*chi**2

##################### QUESTION 2: does the following switch reflect the points in your email?
# Now optimize
#        print 'I''m just before the optimizer loop'
#        for i in range(2):
#            print '-------------------------- optimizer loop i=%s' % str(i+1)
#            sampler = AMAOpt(pars,[logP],[],cov=numpy.array(cov))
#            sampler.sample(Nsample)
        # Switch from optimizer to sampler
        sampler = Sampler(pars,[logP],[])
        sampler.setCov(numpy.array(cov))
        sampler.sample(Nsample)

        # Get results from sampler -- "trace" contains samples from the posterior, calc cov/mean/median on it!
        rslt = sampler.result()

##################### QUESTION 3: can I simply comment out the following part?
        logp,trace,result = sampler.result()
        for i in range(len(pars)):
            pars[i].value = trace[-1][i]

    return rslt


#======================================================================================================================
def getModel(img, noise, lensGal, src, lenses, x, y, agn1, agn2, psf):
    """
    cut from the original built-in subroutine at pylens main function
    """
#    print '--------- enter getModel'
# This evaluates the model at the given parameters and finds the best amplitudes for the surface brightness components
    I = img.flatten()
    S = noise.flatten()
# We need to update the parameters of the objects to the values proposed by the optimizer
    lensGal.setPars()
    src.setPars()
    for l in lenses:
        l.setPars()
    agn1.setPars()
    agn2.setPars()

    xl,yl = pylens.getDeflections(lenses,[x,y])

    model = numpy.empty((img.size,2))
    model[:,0] = convolve.convolve(lensGal.pixeval(x,y),psf,False)[0].ravel()/S
    model[:,1] = convolve.convolve(src.pixeval(xl,yl),psf,False)[0].ravel()/S

    amps,chi = optimize.nnls(model,I/S)
#    print '========= exit getModel'
    return chi,amps*(model.T*S).T


#======================================================================================================================
def run(sys_para, instr_para, ep_para, psf_name, outputFileName, Nsample=1500, flag=True):
    """
    To conduct simulation and MCMC fitting iteratively
    *NOTE: here sys_para.ndim can be any int value besides 1!
    
    === OUTPUT ===
        a file in cPickle(protocol=2) format with the name given by "outputFileName" recording all MCMC fitting results,
        after reading in, it manifests as a tuple with n=sys_para.ndim (including 1!), 
        each element is an n=3 tuple reflecting the MCMC result given by "sampler.result()" in _wrapper_
        
        *NOTE: the current concatenation method of tuples is just a tempory trick, 
               suggestions on more sophisticated method greatly welcome!
    """
    mcmcrslt = ()
    if sys_para.ndim == 1:
        mcmcrslt_new = wrapper(sys_para, instr_para, ep_para, psf_name, Nsample, flag)
        mcmcrslt = mcmcrslt + (mcmcrslt_new,)
        mcmcrslt_new = None

    else:
        Nobj=len(sys_para)
        for i in xrange(Nobj):
            print '#------------------------ begin run No. %s' % str(i+1)
            mcmcrslt_new=wrapper(sys_para[i,:], instr_para, ep_para, psf_name, Nsample, flag)
            print 'simulation and MCMC fitting No. '+str(i+1)+' completed!'
            mcmcrslt = mcmcrslt + (mcmcrslt_new,)
            mcmcrslt_new = None

    # dump the tuple (containing all results) into a pickle in the most compact format (protocol=2)
    f = open(outputFileName,'wb')
    cPickle.dump(mcmcrslt,f,2)
    f.close()

