import lens_fitting
import numpy
import time
t = time.time()

workdir=    '/home/xlmeng/python_scorza2/GitHub_push/'
sys_name=   workdir+'HST/src_lens_HST_para.txt'
instr_name= workdir+'HST/tele_para.txt'
ep_name=    workdir+'HST/ep_para.txt'
psf_name=   workdir+'PSF/HST_F814W.fits'

sys_allpara=numpy.genfromtxt(sys_name,dtype=float,comments='#')
sys_1=sys_allpara[0,:]

instr_allpara=numpy.genfromtxt(instr_name,dtype=float,comments='#')
instr_HST=instr_allpara[0,:]

#ep_allpara=numpy.genfromtxt(ep_name,dtype=float,comments='#')
#ep_50s=ep_allpara[7,:]
ep_all = [10.0, 50.0, 300.0, 1000.0, 5000.0, 10000.0, 50000.0, 300000.0]

# run just 1 line <=> 1 lens system
lens_fitting.run(sys_1, instr_HST, ep_all[1], psf_name, 'HST_50s_iter1.cpkl', Nsample=20, burnin=0.5) 

# run the whole file, i.e., 2 lines  <=>  2 lens systems
#lens_fitting.run(sys_allpara, instr_HST, ep_50s, psf_name, 'temp_optim.dat1', Nsample=2000)

# calc the time elapsed
elapsed = time.time() - t
print 'Elapsed time running the code: %s sec' % str(elapsed)

