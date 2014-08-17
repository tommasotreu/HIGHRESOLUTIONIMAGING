import lens_fitting
import numpy

workdir=    '/home/xlmeng/python_scorza2/GitHub_push/'
sys_name=   workdir+'src_lens_HST_para.txt'
instr_name= workdir+'tele_para.txt'
psf_name=   workdir+'PSF/HST_F814W.fits'

instr_allpara=numpy.genfromtxt(instr_name,dtype=float,comments='#')
instr_HST=instr_allpara[0,:]

lens_fitting.run(sys_name,instr_HST,psf_name)
