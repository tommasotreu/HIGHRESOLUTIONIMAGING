#======================================================================================================================
# IMPORTING MODULES
import numpy
import pylab
import pyfits
import pymc
import sys
import convolve,SBModels
import pylens,MassModels
import indexTricks as iT
import cPickle
from math import log10,ceil
from SampleOpt import AMAOpt,Sampler
from scipy import optimize

#======================================================================================================================
def wrapper(sys_para, instr_para, ep, psf_name, Nsample, burnin, flag):
    """
    This subroutine modulizes the main function "example4xiaolei.py" of the pylens software package   
    sys_para    :   an ndim=1 array containing parameters for the lens+source systems
    instr_para  :   an ndim=1 array containing parameters for the observational instruments (HST, LSST, etc.)
    ep          :   a number giving the exposure time
    psf_name    :   a string leading to the path+name of the psf fits file
    Nsample     :   an int giving the sample size of MCMC fitting
    burnin      :   the number of points to be burnt
    flag        :   a boolean deciding whether to run MCMC ot not
    """
    zp_1s = float(instr_para[2])       
    pix_num = float(instr_para[5])     
    read_val = float(instr_para[3])    
    bg_val = float(instr_para[4])      
    max_exp = float(instr_para[6])     

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
    # create the coordinate grids that the image will be evaluated on
    y,x = iT.coords((pix_num,pix_num))

    #--------------------------------------------------------------------------------------
    # create a PSF   
    psf = pyfits.open(psf_name)[0].data
    psf /= psf.sum()

    #--------------------------------------------------------------------------------------
    # add AGN
    agn1 = SBModels.PointSource('agn1',psf,{'x':sys_para[25],'y':sys_para[26]})
    mag_AGN = sys_para[31]
    AGNMag = agn1.Mag(zp)
    agn1.amp = 10.**(-0.4*(mag_AGN-AGNMag))*sys_para[27]

    agn2 = SBModels.PointSource('agn2',psf,{'x':sys_para[28],'y':sys_para[29]})
    mag_AGN = sys_para[31]
    AGNMag = agn2.Mag(zp)
    agn2.amp = 10.**(-0.4*(mag_AGN-AGNMag))*sys_para[30]

    #--------------------------------------------------------------------------------------    
    # Form the image and convolve (convolution returns the image and FFT'd PSF)
    img,psf = convolve.convolve(lensGal.pixeval(x,y),psf)

    #--------------------------------------------------------------------------------------
    # add AGNs to the image
    img +=agn1.pixeval(x,y)
    img +=agn2.pixeval(x,y)

    xl,yl = pylens.getDeflections(lenses,[x,y])
    img +=convolve.convolve(src.pixeval(xl,yl),psf,False)[0]

    counts = numpy.array(img)

    #--------------------------------------------------------------------------------------    
    #add noise
    read = numpy.zeros((pix_num,pix_num))
    read[:,:] = read_val*read_val*(ceil(ep/max_exp))
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
        pylab.imshow(img,origin='lower',interpolation='nearest')
        pylab.colorbar()
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
        @pymc.observed
        def logP(value=0,tmpXXX=pars): # This is a non-standard use of pymc....
            chi = getModel(img, noise_random, lensGal, src, lenses, x, y, agn1, agn2, psf)[0]
            return -0.5*chi**2

        sampler = Sampler(pars,[logP],[])
        sampler.setCov(numpy.array(cov))
        sampler.sample(Nsample)
        if burnin < 1:
            burnin = int(Nsample * burnin)
        logp,trace,result = sampler.result()
        cov_optm = numpy.cov(trace[burnin:].T)
        for i in range(len(pars)):
            pars[i].value = trace[-1][i]
        sampler = Sampler(pars,[logP],[])
        sampler.setCov(numpy.array(cov_optm))
        sampler.sample(Nsample)
        rslt = sampler.result()

    return rslt


#======================================================================================================================
def getModel(img, noise, lensGal, src, lenses, x, y, agn1, agn2, psf):
    """
    This subroutine calculates the value of chi
    cut from the original built-in subroutine at pylens main function
    """
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
    return chi,amps*(model.T*S).T


#======================================================================================================================
def run(sys_para, instr_para, ep_para, psf_name, outputFileName=None, Nsample=2000, burnin=500, flag=True):
    """
    This subroutine conducts simulation and MCMC fitting iteratively
    OUTPUT: a file with the name given by "outputFileName" recording all MCMC fitting results
    """
    mcmcrslt = ()
    if sys_para.ndim == 1:
        mcmcrslt_new = wrapper(sys_para, instr_para, ep_para, psf_name, Nsample, burnin, flag)
        mcmcrslt = mcmcrslt + (mcmcrslt_new,)
        mcmcrslt_new = None

    else:
        Nobj=len(sys_para)
        for i in xrange(Nobj):
            mcmcrslt_new=wrapper(sys_para[i,:], instr_para, ep_para, psf_name, Nsample, burnin, flag)
            mcmcrslt = mcmcrslt + (mcmcrslt_new,)
            mcmcrslt_new = None

    if outputFileName != None and flag == True:
        f = open(outputFileName,'wb')
        cPickle.dump(mcmcrslt,f,2)
        f.close()

#===================================================================================================
def stats(data):
    data_array = numpy.array(data)
    if data_array.ndim == 1:
        sum = 0.0
        for i in xrange(len(data)):
            sum += data_array[i]
        mean = sum/len(data)
        variance = numpy.sum((data_array-mean)**2.)/len(data)
        stdev = numpy.sqrt(variance)
        return (mean,stdev)
    else:
        print 'the ndim of the file is not N*1!'
        
