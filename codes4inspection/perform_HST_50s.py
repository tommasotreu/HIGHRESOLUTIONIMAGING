import lens_fitting
import numpy

workdir=    '/home/xlmeng/python_scorza2/GitHub_push/'
sys_name=   workdir+'HST/src_lens_HST_para.txt'
instr_name= workdir+'HST/tele_para.txt'
ep_name=    workdir+'HST/ep_para.txt'
psf_name=   workdir+'PSF/HST_F814W.fits'

sys_allpara=numpy.genfromtxt(sys_name,dtype=float,comments='#')
sys_1=sys_allpara[0,:]

instr_allpara=numpy.genfromtxt(instr_name,dtype=float,comments='#')
instr_HST=instr_allpara[0,:]

ep_allpara=numpy.genfromtxt(ep_name,dtype=float,comments='#')
ep_50s=ep_allpara[0,:]

# run just 1 line <=> 1 lens system
#lens_fitting.run(sys_1, instr_HST, ep_50s, psf_name, 'temp_sample.dat1', Nsample=50) 

# run the whole file, i.e., 2 lines  <=>  2 lens systems
lens_fitting.run(sys_allpara, instr_HST, ep_50s, psf_name, 'HST50s.cpikl', Nsample=2000)
