#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import lens_fitting
import numpy
import time
t = time.time()

sys_name=   'src_lens_HST_para.txt'
instr_name= 'teleParam.txt'
#ep_name=    workdir+'Fitting/ep_para.txt'
psf_name=   'HST_F814W.fits'

sys_allpara=numpy.genfromtxt(sys_name,dtype=float,comments='#')
sys_chosen=sys_allpara[0:1,:]

# get all data (including telescope names) in string format 
instr_allpara  = numpy.genfromtxt(instr_name,dtype=str,comments='#')
instr_chosen= instr_allpara[0,:]
#<<<140826>>>NOTE: currently you can only calc instrument one by one, but no need to specify its name by yourself

#ep_allpara=numpy.genfromtxt(ep_name,dtype=float,comments='#')
#ep_50s=ep_allpara[7,:]
ep = [7500.0, 2500.0, 750.0, 250.0, 75.0, 50.0, 150.0, 450.0]
ep_chosen = ep[6:7]

# number of chosen exposure times
Nep = numpy.array(ep_chosen).size

# number of iterations
Niter = 1

# MCMC sample parameters
Nsample = 20000
burnin  = 0.2

# times of the cov
Ncov = 0.01

# execute pylens
print '============== Start doing telescope %s at filter %s' % (instr_chosen[0],instr_chosen[1])
for i in xrange(Nep):
    print '---------- Start doing exposure time = %s sec' % str(ep_chosen[i])
    for j in xrange(Niter):
        outputFileName = instr_chosen[0]+'_'+instr_chosen[1]+'_'+str(int(ep_chosen[i]))+'s_iter'+str(j+1)+'.cpkl'
        print 'Iteration No. ', j+1, 'output file name =', outputFileName
        lens_fitting.run(sys_chosen, instr_chosen, ep_chosen[i], psf_name, outputFileName, Nsample, Ncov, burnin, flag=False)

# calc the time elapsed
elapsed = time.time() - t
print 'Elapsed time running the code: %s sec' % str(elapsed)

