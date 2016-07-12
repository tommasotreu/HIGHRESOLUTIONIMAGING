import numpy,time
from math import pi,cos as COS,sin as SIN,asin as ASIN
"""
convert Adri's mathematica code about a bunch of functions~

by Xiao-Lei Meng:)
"""

"""
A bunch of functions, you will Not need the Sersic ones, i.e. Sersic or flase. The others, in particular Gint and flatrot, are the core of the module. Gint is a Gaussian, with widths dx and dy in pixels, concered on (0,0). flatrot takes two position coordinates (x,y), plus an position angular 'pa', and a flattening 'flat', computes the rotated elliptical radius. You'll need them when placing Gaussian blobs on top of images.
"""

#parameter n describes the 'shape' of the light-profile
def bs(n):
    bs = 2.*n-1./3.
    return(bs)

#precise bs
def bs_pc(n):
    bs_pc = 2.*n-1./3.+4./(405.*n)+46./(25515.*n**2.)
    return(bs_pc)

#question: where is re (r_effective)?
def Sersic(R,n):
    Sersic = numpy.exp(-1.*bs(n)*(R**(1./n)))
    return(Sersic)

#here, flat is the axis ratio
def flase(x,y,flat,pa,n):
    flase = Sersic((flat*(x*COS(pa)+y*SIN(pa))**2.+(y*COS(pa)-x*SIN(pa))**2./flat)**0.5,n)
    return(flase)
    
#this is a cylindrical Gaussian in 2D radius x, note the denominator!
def G(x,dx):
    G = (numpy.exp(-0.5*x**2./(dx**2.)))/(2.*pi*(dx**2.))
    return(G)

#this is a 2D Gaussian averaged over small pixels;3x3 grids.
def Gint(x,y,dx,dy):
    return (9./16.)*G(x,dx)*G(y,dy)+(3./32.)*(G(x+1.,dx)*G(y,dy)+G(x-1.,dx)*G(y,dy)+\
        G(x,dx)*G(y+1.,dy)+G(x,dx)*G(y-1.,dy))+(1./64.)*(G(x-1.,dx)*G(y-1.,dy)+\
        G(x-1.,dx)*G(y+1.,dy)+G(x+1.,dx)*G(y-1.,dy)+G(x+1.,dx)*G(y+1.,dy))
#    return(Gint)

#this computes the elliptical radius
def flatrot(x,y,flat,pa):
    flatrot = (flat*(x*COS(pa)+y*SIN(pa))**2.+(y*COS(pa)-x*SIN(pa))**2./flat)**0.5
    return(flatrot)
#e.g. a Gaussian component will be G((flatrot(#1-xcenter,#2-ycenter,flat,pa)),sigma), where #1 and #2 are x and y indices.

#translate Iij second moments into shape parameters?
def getshapes(Ixx,Iyy,Ixy):
    Delta = ((Ixx-Iyy)**2.+4.*Ixy**2.)**0.5
    ftmp = ((Ixx+Iyy-Delta)/(Ixx+Iyy+Delta))**0.5
    phitmp = ASIN(2.*Ixy/Delta)/2.
    phitmp = numpy.greater_equal(Ixx,Iyy)*phitmp+numpy.less(Ixx,Iyy)*(numpy.sign(Ixy)*pi/2.-phitmp)
    phitmp = phitmp%3.14159
    Rtmp = (Ixx+Iyy)**0.5
    return(Rtmp,phitmp,ftmp)

#partry<--->Join (centers,secmoms,strehl)
#build nof "model" PSFs in each band, without flux normalisation, return it
#e.g. a Gaussian component will be G(flatrot(#1-xcenter,#2-ycenter,flat,pa),sigma), where #1 and #2 are x and y indices
#def paintpsf(numx,numy,num3,):
#    cores = 
#   Array[G[flatrot[#2 - xcenter[[#1]], #3 - ycenter[[#1]], 
#       innflas[[#1]], innpas[[#1]]], innsigmas[[#1]]] &, {nof, imsize,
#      imsize}]; 
#  wings = Array[
#    G[flatrot[#2 - xcenter[[#1]], #3 - ycenter[[#1]], outflas[[#1]], 
#       outpas[[#1]]], outsigmas[[#1]]] &, {nof, imsize, imsize}]; 
#  tottot = Array[Total[Total[cores[[#]]]]/Total[Total[wings[[#]]]] &, 
#    nof];
#  fluxratio = strenl^(-1) - 1;
#  fluxratio = fluxratio/tottot;
#  modelpsf = Array[cores[[#]] + fluxratio[[#]]*wings[[#]] &, nof];
#  modelpsf = Array[modelpsf[[#]]/Total[Total[modelpsf[[#]]]] &, nof];)









