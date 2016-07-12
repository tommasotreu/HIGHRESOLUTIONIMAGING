import pyfits
import numpy as np
import pylab
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import math
import pdb
import function
import commands
from scipy.misc import imsave
import aplpy
from pyavm import AVM

#the following part has benn checked.
#--------------------------
#the following lines are used to load the cutouts of a target with name 3038925090_?.fits
datapath = '/home/xlmeng/my_works/fitting_pointsource_to_lensimagecandidate/writtenby_me/dependent/Image/'
idd = '3038925090'
nameg = idd+'_g.fits'
namer = idd+'_r.fits'
namei = idd+'_i.fits'
namez = idd+'_z.fits'
nameY = idd+'_Y.fits'

#select hdulist[0] or hdulist[1] according to your data type 
hdulist1 = pyfits.open(datapath+nameg)
imgg = hdulist1[0].data
#imgg = hdulist1[1].data
print imgg.shape
hdulist2 = pyfits.open(datapath+namer)
imgr = hdulist2[0].data
#imgr = hdulist2[1].data
hdulist3 = pyfits.open(datapath+namei)
imgi = hdulist3[0].data
#imgi = hdulist3[1].data
hdulist4 = pyfits.open(datapath+namez)
imgz = hdulist4[0].data
#imgz = hdulist4[1].data
hdulist5 = pyfits.open(datapath+nameY)
imgY = hdulist5[0].data
#imgY = hdulist5[1].data
##---------------------------------------------------

#-------------------------------
#this is the length of each side of each DES cutout
imsize = len(imgg)
red = 1.25*imgi/3.
green = 1.25*imgr/2.5
blue = 1.25*imgg

a=red[::-1]
b=green[::-1]
c=blue[::-1]

#n is the dimension.
n=3
rgb=np.asarray(zip(a.flatten(),b.flatten(),c.flatten())).reshape(a.shape[0],a.shape[1],n)

#---------------------------------
#pylab.imshow(a,origin='lower',interpolation='nearest')
#pylab.imshow(a,origin='upper',interpolation='nearest')
#pylab.colorbar()
#pylab.show()
#pylab.savefig('red.png',dpi=200)
#pyfits.PrimaryHDU(a).writeto('red.fits',clobber=True)
##----------------------------------------------------

#this is needed in case you want to save the cutouts in a pdf file
#--------------------------------
outname = 'DES'+idd+'.pdf'
##----------------------------------------------------

#this packs the grizY cutouts in a "file" array,of length "nof".
#In python, you'd need to initialize a nof-size-size array,
#where "size" is the length of each side of a cutout
#----------------------------------
emptylist=[]
emptylist.append(imgg)
emptylist.append(imgr)
emptylist.append(imgi)
emptylist.append(imgz)
emptylist.append(imgY)
file = np.asarray(emptylist)

nof = len(file)
imsizes = np.linspace(file.shape[1],file.shape[1],num=nof)
##-----------------------------------------------------

#this cuts the central part of the grizY cutouts.
#To be on the safe side, I'd recommend replacing "cuts" with 1 and "cutw" wth imsizes[[1]]
#math.floor(x). Return the floor of x, the largest integer less than or equal to x. If x is not a float, delegates to x.__floor__(), which should return an Integral value.
#---------------------------------
#cuts = math.floor(imsizes[1]/4.)
#cutw = math.floor(imsizes[1]/2.)
cuts = 1.
cutw = math.floor(imsizes[1])


file = file[:,cuts-1:cuts+cutw-1,cuts-1:cuts+cutw-1]
imsizes = np.linspace(file.shape[1],file.shape[1],num=nof)

#This is the pixel size in arcseconds for DECamera
pixsizes = np.linspace(0.263,0.263,num=nof)
##------------------------------------------------------

#the following is a QnD way of estimating sky brightness and noise in each of the grizY cutouts, you should actually use a median or a trimmed mean (google them!).
#You'll need the backg and bcknoise in the Em routine to place point sources,excluding regions that have low S/N. Here skypix,backg, bcknoise are arrays of length "nof".
#----------------------------------
fovs = imsizes/4.
i = np.arange(1,imsizes[1]+1)

step = map(lambda x:(x-math.floor(imsizes[1]/2))**2+(x-math.floor(imsizes[1]/2))**2-(fovs[0])**2,i)
step = np.asarray(step)
step[step>=0]=1
step[step<0]=0
skypix = np.linspace(sum(step)*imsizes[1],sum(step)*imsizes[1],num=nof)

backg = np.zeros(nof)
for zz in xrange(nof):
    for ii in xrange(int(imsizes[1])):
        backg[zz] += step[ii]*np.sum(file[zz,ii,:])/skypix[zz]

bcknoise_in = np.zeros((nof,imsizes[1],imsizes[1]))
for zz in xrange(nof):
    bcknoise_in[zz,:,:] = backg[zz]

filebackg = np.zeros((nof,imsizes[1],imsizes[1]))
filebackg = (file-bcknoise_in)**2.

bcknoisesum = np.zeros(nof)
for zz in xrange(nof):
    for ii in xrange(int(imsizes[1])):
        bcknoisesum[zz] += step[ii]*np.sum(filebackg[zz,ii,:])/skypix[zz]
        bcknoise = (bcknoisesum)**0.5
##--------------------------------------------------------

#from the raw data cutouts above,build the "smoothed" data and an estimate of the noise maps. To do so, you'll use a convolution with a 3-by-3 smoothing kernel "smint".
#Adriano chose kernel shape
#-----------------------------------
smint = np.zeros((3,3))
smint = np.array([[1./64., 3./32., 1./64.],[3./32.,9./16.,3./32.],[1./64.,3./32.,1./64.]])

delta = np.zeros((3,3))
delta = np.array([[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]])

backg_tran = np.zeros((nof,imsizes[1],imsizes[1]))
for zz in xrange(nof):
    backg_tran[zz,:,:] = backg[zz]

oo=file-backg_tran
data = np.zeros((nof,imsizes[1]-2.,imsizes[1]-2.))
from scipy import signal
for zz in xrange(nof):
    data[zz] = signal.convolve2d(oo[zz,:,:],smint,boundary='wrap',mode='valid')

imsizes = np.linspace(data.shape[1],data.shape[1],num=nof)

derr = np.zeros((nof,data.shape[1],data.shape[2]))
for zz in xrange(nof):
    derr[zz] = signal.convolve2d((oo[zz,:,:])**2.,smint,boundary='wrap',mode='valid')

derr = derr-(data)**2.
derr = (derr)**0.5
#print 'derr=',derr
print 'derr[0,:,:]=',derr[0,:,:]
#print 'derr.shape=',derr.shape
imsize = imsizes[0]

rgb_in1 = (1.25/3.)*data[2,:,:][::-1]
rgb_in2 = (1.25/2.5)*data[1,:,:][::-1]
rgb_in3 = 1.25*data[0,:,:][::-1]
n_rgb = 3.
#here rgb is a 3D data
rgb = np.asarray(zip(rgb_in1.flatten(),rgb_in2.flatten(),rgb_in3.flatten())).reshape(rgb_in1.shape[0],rgb_in1.shape[1],n_rgb)
#print 'rgb=',rgb
print 'rgb.shape=',rgb.shape

rgb2_in1 = (1.25/3.)*data[3,:,:][::-1]
rgb2_in2 = (1.25/2.5)*data[2,:,:][::-1]
rgb2_in3 = 1.25*data[1,:,:][::-1]
n_rgb2 = 3.
rgb2 = np.asarray(zip(rgb2_in1.flatten(),rgb2_in2.flatten(),rgb2_in3.flatten())).reshape(rgb2_in1.shape[0],rgb2_in1.shape[1],n_rgb2)
#print 'rgb2=',rgb2
print 'rgb2.shape=',rgb2.shape

rgb3_in1 = (1.25/3.)*data[4,:,:][::-1]
rgb3_in2 = (1.25/2.5)*data[3,:,:][::-1]
rgb3_in3 = 1.25*data[2,:,:][::-1]
n_rgb3 = 3.
rgb3 = np.asarray(zip(rgb3_in1.flatten(),rgb3_in2.flatten(),rgb3_in3.flatten())).reshape(rgb3_in1.shape[0],rgb3_in1.shape[1],n_rgb3)

rgbres1_in1 = (1.25/3.)*derr[2,:,:][::-1]
rgbres1_in2 = (1.25/2.5)*derr[1,:,:][::-1]
rgbres1_in3 = 1.25*derr[0,:,:][::-1]
n_rgbres1 = 3.
rgbres1 = np.asarray(zip(rgbres1_in1.flatten(),rgbres1_in2.flatten(),rgbres1_in3.flatten())).reshape(rgbres1_in1.shape[0],rgbres1_in1.shape[1],n_rgbres1)
print 'rgbres1.shape=',rgbres1.shape

rgbres2_in1 = (1.25/3.)*derr[3,:,:][::-1]
rgbres2_in2 = (1.25/2.5)*derr[2,:,:][::-1]
rgbres2_in3 = 1.25*derr[1,:,:][::-1]
n_rgbres2 = 3.
rgbres2 = np.asarray(zip(rgbres2_in1.flatten(),rgbres2_in2.flatten(),rgbres2_in3.flatten())).reshape(rgbres2_in1.shape[0],rgbres2_in1.shape[1],n_rgbres2)

rgbres3_in1 = (1.25/3.)*derr[4,:,:][::-1]
rgbres3_in2 = (1.25/2.5)*derr[3,:,:][::-1]
rgbres3_in3 = 1.25*derr[2,:,:][::-1]
n_rgbres3 = 3.
rgbres3 = np.asarray(zip(rgbres3_in1.flatten(),rgbres3_in2.flatten(),rgbres3_in3.flatten())).reshape(rgbres3_in1.shape[0],rgbres3_in1.shape[1],n_rgbres3)

signoise = np.zeros((nof,data.shape[1],data.shape[2]))
signoise = np.abs(data/derr)
print 'signoise=',signoise
print 'signoise.shape=',signoise.shape

rgbsn1_in1 = (1.25/3.)*signoise[2,:,:][::-1]
rgbsn1_in2 = (1.25/2.5)*signoise[1,:,:][::-1]
rgbsn1_in3 = 1.25*signoise[0,:,:][::-1]
n_rgbsn1 = 3.
rgbsn1 = np.asarray(zip(rgbsn1_in1.flatten(),rgbsn1_in2.flatten(),rgbsn1_in3.flatten())).reshape(rgbsn1_in1.shape[0],rgbsn1_in1.shape[1],n_rgbsn1)
print 'rgbsn1.shape=',rgbsn1.shape

rgbsn2_in1 = (1.25/3.)*signoise[4,:,:][::-1]
rgbsn2_in2 = (1.25/2.5)*signoise[3,:,:][::-1]
rgbsn2_in3 = 1.25*signoise[2,:,:][::-1]
n_rgbsn2 = 3.
rgbsn2 = np.asarray(zip(rgbsn2_in1.flatten(),rgbsn2_in2.flatten(),rgbsn2_in3.flatten())).reshape(rgbsn2_in1.shape[0],rgbsn2_in1.shape[1],n_rgbsn2)

rgbsn3_in1 = (1.25/3.)*signoise[3,:,:][::-1]
rgbsn3_in2 = (1.25/2.5)*signoise[2,:,:][::-1]
rgbsn3_in3 = 1.25*signoise[1,:,:][::-1]
n_rgbsn3 = 3.
rgbsn3 = np.asarray(zip(rgbsn3_in1.flatten(),rgbsn3_in2.flatten(),rgbsn3_in3.flatten())).reshape(rgbsn3_in1.shape[0],rgbsn3_in1.shape[1],n_rgbsn3)

#put a lower level to the estimated noise, when it's too low
ifi = 0.
bcknoise_usehere = np.zeros((nof,imsizes[1],imsizes[1]))
for zz in xrange(nof):
    bcknoise_usehere[zz,:,:] = bcknoise[zz]
while ifi<=(nof-1.):
    derr[ifi,:,:] = np.maximum(derr[ifi,:,:],bcknoise_usehere[ifi,:,:])
    ifi = ifi+1.
print 'derr.shape=',derr.shape
#print 'derr=',derr
##-------------------------------------------------

#the first plot that we should build: the first row has the data in grizY bands, plus (g,r,i), (r,i,z), (i,z,Y) colour-composites. The second row is the same but fr the estimated noise. The third one is for the signal-to-noise ratio. I've used coefficients that let you see something in the colour-composites, feelfree to choose your own.
#--------------------------------------
rgb_band_root = 'rgb_band'
rgb_band = np.zeros((imsizes[1],imsizes[1]))
for zz in xrange(nof-2):
    rgb_band = rgb[:,:,zz]/50.
    rgb_band_name = idd+rgb_band_root+str(zz+1)+'.fits'
    pyfits.PrimaryHDU(rgb_band).writeto(rgb_band_name,clobber=True)

#If you are starting from three images with different projections/resolutions, the first step is to reproject these to a common projection/resolution using the make_rgb_cube() function.
#aplpy.make_rgb_cube(['rgb_band1.fits','rgb_band2.fits','rgb_band3.fits'],'rgb_cube.fits')

#The make_rgb_image() function can be used to produce an RGB image from either three FITS files in the exact same projection, or a FITS cube containing the three channels (such as that output by make_rgb_cube()).
#aplpy.make_rgb_image([idd+'rgb_band1.fits',idd+'rgb_band2.fits',idd+'rgb_band3.fits'],idd+'combine_rgb.png')


rgb2_band_root = 'rgb2_band'
rgb2_band = np.zeros((imsizes[1],imsizes[1]))
for zz in xrange(nof-2):
    rgb2_band = rgb2[:,:,zz]/50.
    rgb2_band_name = idd+rgb2_band_root+str(zz+1)+'.fits'
    pyfits.PrimaryHDU(rgb2_band).writeto(rgb2_band_name,clobber=True)
#aplpy.make_rgb_image([idd+'rgb2_band1.fits',idd+'rgb2_band2.fits',idd+'rgb2_band3.fits'],idd+'combine_rgb2.png')

rgb3_band_root = 'rgb3_band'
rgb3_band = np.zeros((imsizes[1],imsizes[1]))
for zz in xrange(nof-2):
    rgb3_band = rgb3[:,:,zz]/50.
    rgb3_band_name = idd+rgb3_band_root+str(zz+1)+'.fits'
    pyfits.PrimaryHDU(rgb3_band).writeto(rgb3_band_name,clobber=True)
#aplpy.make_rgb_image([idd+'rgb3_band1.fits',idd+'rgb3_band2.fits',idd+'rgb3_band3.fits'],idd+'combine_rgb3.png')

derr_band_root = 'derr_band'
derr_band = np.zeros((imsizes[1],imsizes[1]))
for zz in xrange(nof):
    derr_band = 20.*derr[zz,:,:]
    derr_band_name = idd+derr_band_root+str(zz+1)+'.fits'
    pyfits.PrimaryHDU(derr_band).writeto(derr_band_name,clobber=True)


rgbres1_band_root = 'rgbres1_band'
rgbres1_band = np.zeros((imsizes[1],imsizes[1]))
for zz in xrange(nof-2):
    rgbres1_band = rgbres1[:,:,zz]/50.
    rgbres1_band_name = idd+rgbres1_band_root+str(zz+1)+'.fits'
    pyfits.PrimaryHDU(rgbres1_band).writeto(rgbres1_band_name,clobber=True)
#aplpy.make_rgb_image([idd+'rgbres1_band1.fits',idd+'rgbres1_band2.fits',idd+'rgbres1_band3.fits'],idd+'combine_rgbres1.png')

rgbres2_band_root = 'rgbres2_band'
rgbres2_band = np.zeros((imsizes[1],imsizes[1]))
for zz in xrange(nof-2):
    rgbres2_band = rgbres2[:,:,zz]/50.
    rgbres2_band_name = idd+rgbres2_band_root+str(zz+1)+'.fits'
    pyfits.PrimaryHDU(rgbres2_band).writeto(rgbres2_band_name,clobber=True)
#aplpy.make_rgb_image([idd+'rgbres2_band1.fits',idd+'rgbres2_band2.fits',idd+'rgbres2_band3.fits'],idd+'combine_rgbres2.png')

rgbres3_band_root = 'rgbres3_band'
rgbres3_band = np.zeros((imsizes[1],imsizes[1]))
for zz in xrange(nof-2):
    rgbres3_band = rgbres3[:,:,zz]/50.
    rgbres3_band_name = idd+rgbres3_band_root+str(zz+1)+'.fits'
    pyfits.PrimaryHDU(rgbres3_band).writeto(rgbres3_band_name,clobber=True)
#aplpy.make_rgb_image([idd+'rgbres3_band1.fits',idd+'rgbres3_band2.fits',idd+'rgbres3_band3.fits'],idd+'combine_rgbres3.png')

signoise_band_root = 'signoise_band'
signoise_band = np.zeros((imsizes[1],imsizes[1]))
for zz in xrange(nof):
    signoise_band = signoise[zz,:,:]
#    print 'signoise_band=',signoise_band
    signoise_band_name = idd+signoise_band_root+str(zz+1)+'.fits'
    pyfits.PrimaryHDU(signoise_band).writeto(signoise_band_name,clobber=True)
#    print 'signoise_band_name=',signoise_band_name

rgbsn1_band_root = 'rgbsn1_band'
rgbsn1_band = np.zeros((imsizes[1],imsizes[1]))
for zz in xrange(nof-2):
    rgbsn1_band = rgbsn1[:,:,zz]/20.
    rgbsn1_band_name = idd+rgbsn1_band_root+str(zz+1)+'.fits'
    pyfits.PrimaryHDU(rgbsn1_band).writeto(rgbsn1_band_name,clobber=True)
#aplpy.make_rgb_image([idd+'rgbsn1_band1.fits',idd+'rgbsn1_band2.fits',idd+'rgbsn1_band3.fits'],idd+'combine_rgbsn1.png')

rgbsn2_band_root = 'rgbsn2_band'
rgbsn2_band = np.zeros((imsizes[1],imsizes[1]))
for zz in xrange(nof-2):
    rgbsn2_band = rgbsn2[:,:,zz]/20.
    rgbsn2_band_name = idd+rgbsn2_band_root+str(zz+1)+'.fits'
    pyfits.PrimaryHDU(rgbsn2_band).writeto(rgbsn2_band_name,clobber=True)
#aplpy.make_rgb_image([idd+'rgbsn2_band1.fits',idd+'rgbsn2_band2.fits',idd+'rgbsn2_band3.fits'],idd+'combine_rgbsn2.png')

rgbsn3_band_root = 'rgbsn3_band'
rgbsn3_band = np.zeros((imsizes[1],imsizes[1]))
for zz in xrange(nof-2):
    rgbsn3_band = rgbsn3[:,:,zz]/20.
    rgbsn3_band_name = idd+rgbsn3_band_root+str(zz+1)+'.fits'
    pyfits.PrimaryHDU(rgbsn3_band).writeto(rgbsn3_band_name,clobber=True)
#aplpy.make_rgb_image([idd+'rgbsn3_band1.fits',idd+'rgbsn3_band2.fits',idd+'rgbsn3_band3.fits'],idd+'combine_rgbsn3.png')


#some quick flux calibrations: in DES, the magnitudes are as simple as in "bandmags",which again is an array of length "nof"
#----------------------------------
bandmags = np.zeros((nof,1))
for zz in xrange(nof):
    bandmags[zz] += -2.5*(np.log(np.sum(data[zz,:,:]))/np.log(10.)-9.)
#if the value in the log is minus, the results show 'nan'. Don't worry, that becuse the file is simulated, not real.

minmags = bandmags-0.5
maxmags = bandmags+2.5*1.
minflux = 10.**(9.-0.4*maxmags)
maxflux = 10.**(9.-0.4*minmags)
minflux = 3.*bcknoise
##-----------------------------------------------

#now do the 1PSF fit
#one Gaussian fit
#-----------------------------------
centroids = np.zeros((nof,2))
for xx in xrange(nof):
    centroids[xx,:] = imsizes[1]/2.

Iner_in = np.zeros((2,2))
Iner_in = np.array([[3.**2.,0.],[0.,3.**2.]])
Iner = np.zeros((nof,2,2))
for zz in xrange(nof):
    Iner[zz] = Iner_in[:,:]

irec = 1.
Nrecpsf = 2.

#np.linalg.pinv is used to get the pseudoinverse of Iner, but what's the effect of it? I don't know...
Miner = np.zeros((nof,2,2))
for zz in xrange(nof):
    Miner[zz] = np.linalg.pinv(Iner[zz,:,:])
##-----------------------------------------------

#if you want to avoid summing over all pixels, put some convenient limits
#there are problems when I run the code with constant 10; I found the reason: the fits I used haven't enough pixels, so when + or - 10 they beyond the edge. So I suggest that using a variate (e.g. imsizes[1]/5. instead of the constant.
#----------------------------------
#kilo = np.linspace(math.floor(centroids[:,0][0])-10.,math.floor(centroids[:,0][0])-10.,num=nof)
kilo = np.linspace(math.floor(centroids[:,0][0])-5.,math.floor(centroids[:,0][0])-5.,num=nof)
#kihi = np.linspace(math.floor(centroids[:,0][0])+10.,math.floor(centroids[:,0][0])+10.,num=nof)
kihi = np.linspace(math.floor(centroids[:,0][0])+5.,math.floor(centroids[: ,0][0])+5.,num=nof)
#kjlo = np.linspace(math.floor(centroids[:,1][0])-10.,math.floor(centroids[:,1][0])-10.,num=nof)
kjlo = np.linspace(math.floor(centroids[:,1][0])-5.,math.floor(centroids[: ,1][0])-5.,num=nof)
#kjhi = np.linspace(math.floor(centroids[:,1][0])+10.,math.floor(centroids[:,1][0])+10.,num=nof)
kjhi = np.linspace(math.floor(centroids[:,1][0])+5.,math.floor(centroids[: ,1][0])+5.,num=nof)
##---------------------------------------------

#adaptive moments below need the prefctor 2 because of ws in the sum
#Gaussian windowing function
#---------------------------------
ws = np.zeros((nof,imsizes[1],imsizes[1]))
for zz in xrange(nof):
    for ii in xrange(int(imsizes[1])):
        for jj in xrange(int(imsizes[1])):
            ws[zz,ii,jj] = math.exp(-0.5*Miner[zz,0,0]*(ii+1.-centroids[zz,0])**2.-0.5*Miner[zz,1,1]*(jj+1.-centroids[zz,1])**2.-Miner[zz,0,1]*(ii+1.-centroids[zz,0])*(jj+1.-centroids[zz,1]))

step_data = np.zeros((nof,imsizes[1],imsizes[1]))
step_data[:] = data
step_data[step_data>=0]=1
step_data[step_data<0]=0
psfws = np.zeros((nof,imsizes[1],imsizes[1]))
for zz in xrange(nof):
    for ii in xrange(int(imsizes[1])):
        for jj in xrange(int(imsizes[1])):
            psfws[zz,ii,jj] = data[zz,ii,jj]*ws[zz,ii,jj]*step_data[zz,ii,jj]

nws = np.zeros((nof,1))
for zz in xrange(nof):
    for ii in xrange(int(imsizes[1])):
        for jj in xrange(int(imsizes[1])):
            nws[zz] += np.sum(psfws[zz,ii,jj])

kj_matr = np.zeros((nof,int(kjhi[1]-kjlo[1]+1.),int(kjhi[1]-kjlo[1]+1.)))
kj_matrin = np.zeros((int(kjhi[1]-kjlo[1]+1.),int(kjhi[1]-kjlo[1]+1.)))
for ii in xrange(int(kjhi[1]-kjlo[1]+1.)):
    kj_matrin[ii] = np.array(np.linspace(int(kjlo[1]),int(kjhi[1]),int(kjhi[1]-kjlo[1]+1.)))
for zz in xrange(nof):
    kj_matr[zz] = kj_matrin[:,:]

ki_matr = np.zeros((nof,int(kihi[1]-kilo[1]+1.),int(kihi[1]-kilo[1]+1.)))
ki_matrin = (np.arange(kilo[1],kihi[1]+1.,1).reshape(-1,1)).repeat(kihi[1]-kilo[1]+1.,axis=1)
for zz in xrange(nof):
    ki_matr[zz] = ki_matrin[:,:]

psfws_in = np.zeros((nof,int(kihi[1]-kilo[1]+1.),int(kihi[1]-kilo[1]+1.)))
for zz in xrange(nof):
    psfws_in[zz] = psfws[zz,(kilo[1]-1):kihi[1],(kjlo[1]-1):kjhi[1]]

cen_xin = np.zeros((nof,int(kihi[1]-kilo[1]+1.),int(kihi[1]-kilo[1]+1.)))
cen_xin = ki_matr*psfws_in
cen_x = np.zeros((nof,1))
for zz in xrange(nof):
    for ii in xrange(int(kihi[1]-kilo[1]+1.)):
        cen_x[zz] += sum(cen_xin[zz,ii,:])/nws[zz]

cen_yin = np.zeros((nof,int(kjhi[1]-kjlo[1]+1.),int(kjhi[1]-kjlo[1]+1.)))
cen_yin = kj_matr*psfws_in
cen_y = np.zeros((nof,1))
for zz in xrange(nof):
    for ii in xrange(int(kjhi[1]-kjlo[1]+1.)):
        cen_y[zz] += sum(cen_yin[zz,ii,:])/nws[zz]

centroids = np.zeros((nof,2))
for zz in xrange(nof):
    centroids[zz] = np.hstack([cen_x[zz],cen_y[zz]])
print 'centroids=',centroids

Iner_11 = np.zeros((nof,int(kihi[1]-kilo[1]+1.),int(kihi[1]-kilo[1]+1.)))
for zz in xrange(nof):
    Iner_11[zz] = (ki_matr[zz,:,:]-centroids[zz,0])**2.

Iner_12 = np.zeros((nof,int(kihi[1]-kilo[1]+1.),int(kihi[1]-kilo[1]+1.)))
for zz in xrange(nof):
    Iner_12[zz] = (ki_matr[zz,:,:]-centroids[zz,0])*(kj_matr[zz,:,:]-centroids[zz,1])

Iner_21 = np.zeros((nof,int(kihi[1]-kilo[1]+1.),int(kihi[1]-kilo[1]+1.)))
for zz in xrange(nof):
    Iner_21[zz] = (ki_matr[zz,:,:]-centroids[zz,0])*(kj_matr[zz,:,:]-centroids[zz,1])

Iner_22 = np.zeros((nof,int(kihi[1]-kilo[1]+1.),int(kihi[1]-kilo[1]+1.)))
for zz in xrange(nof):
    Iner_22[zz] = (kj_matr[zz,:,:]-centroids[zz,1])**2.

Iner_xin1 = np.zeros((nof,int(kihi[1]-kilo[1]+1.),int(kihi[1]-kilo[1]+1.)))
Iner_xin1 = Iner_11*psfws_in
Iner_yin1 = np.zeros((nof,int(kjhi[1]-kjlo[1]+1.),int(kjhi[1]-kjlo[1]+1.)))
Iner_yin1 = Iner_12*psfws_in
Iner_xin2 = np.zeros((nof,int(kihi[1]-kilo[1]+1.),int(kihi[1]-kilo[1]+1.)))
Iner_xin2 = Iner_21*psfws_in
Iner_yin2 = np.zeros((nof,int(kjhi[1]-kjlo[1]+1.),int(kjhi[1]-kjlo[1]+1.)))
Iner_yin2 = Iner_22*psfws_in
Iner_x1=np.zeros((nof,1))
Iner_y1 = np.zeros((nof,1))
Iner_x2 = np.zeros((nof,1))
Iner_y2 = np.zeros((nof,1))
for zz in xrange(nof):
    for ii in xrange(int(kihi[1]-kilo[1]+1.)):
        Iner_x1[zz] += 2.*sum(Iner_xin1[zz,ii,:])/nws[zz]
        Iner_y1[zz] += 2.*sum(Iner_yin1[zz,ii,:])/nws[zz]
        Iner_x2[zz] += 2.*sum(Iner_xin2[zz,ii,:])/nws[zz]
        Iner_y2[zz] += 2.*sum(Iner_yin2[zz,ii,:])/nws[zz]

Iner = np.zeros((nof,2,2))
Iner1 = np.zeros((nof,2))
Iner2 = np.zeros((nof,2))
for zz in xrange(nof):
    Iner1[zz] = np.hstack([Iner_x1[zz],Iner_y1[zz]])
    Iner2[zz] = np.hstack([Iner_x2[zz],Iner_y2[zz]])
    Iner[zz] = np.vstack([Iner1[zz],Iner2[zz]])
print 'Iner=',Iner
##-------------------------------------------------

#now adjust the centroids and "Iner" matrices recursively
#---------------------------------

##------------------------------------------------


#these will be used to initialize the 2PSF fit: they give the width of the blob in the grizY bands. Again,"psfsigmas" has length "nof"
psfsigmas = np.zeros((nof,1))
for zz in xrange(nof):
    psfsigmas[zz] = (Iner[zz,0,0]+Iner[zz,1,1])**0.5*1.4142/2.
#print 'psfsigmas=',psfsigmas


#these give you a width (twice as that),a p.a. and a b/a estimate for the blob in each band
pars1psf = np.zeros((nof,3))
for zz in xrange(nof):
    pars1psf[zz] = function.getshapes(Iner[zz,0,0],Iner[zz,1,1],Iner[zz,0,1])
#print 'pars1psf=',pars1psf


#build "nof" fake PSF's, which will be the blobs we'll use to find the best chi^2 with one extended source. Originally I was fitting for core, wings and strehl; here,we'll use basically the same parameters for core and wings, to do it quickly
Rcorep = 0.99*pars1psf[:,0]*0.71
Rwingp = 1.01*pars1psf[:,0]*0.71
pacorep = pars1psf[:,1]
pawingp = pars1psf[:,1]
bawingp = pars1psf[:,2]
bacorep = pars1psf[:,2]
strehlp = np.linspace(0.501,0.501,num=nof)
psfcore = np.zeros((nof,imsizes[1],imsizes[1]))
psfwing = psfcore
magicwin = 5.
#this regulates how many pixels you actually want to paint, using the "kilo"..."kjhi" limits below for each cutout











