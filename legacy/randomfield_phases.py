
import numpy as np
import scipy as sc
from scipy.stats import multivariate_normal
from scipy.signal import convolve as convolve_sc
import matplotlib.pyplot as plt


n = 100
randfield = np.random.rand(n,n)

def convert_to_rhombus(x,y):
    return x+0.5*y,np.sqrt(3)/2*y

def circdiff(vec1,vec2):
    return np.array([np.mod(vec1-vec2, 2*np.pi), np.mod(vec2-vec1, 2*np.pi)]).min(axis=0)

def circmean(vec):
    return np.mod(np.angle( sum(np.exp(1j*vec)) / len(vec)), 2*np.pi)

def pairwise_distance(mat,gridx,gridy):
    mat_lin = np.reshape(mat,(1,-1))[0]
    m1,m2 = np.meshgrid(mat_lin,mat_lin)
    xlin = np.reshape(gridx,(1,-1))[0]
    x1,x2 = np.meshgrid(xlin,xlin)
    ylin = np.reshape(gridy,(1,-1))[0]
    y1,y2 = np.meshgrid(ylin,ylin)
    xdiff = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2)
    phdiff = circdiff(m1,m2)
    
    dx = 5
    bins = np.arange(0,80,dx)
    meandiff = np.zeros(len(bins)-1)
    stddiff = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
       meandiff[i] = np.mean( phdiff[(xdiff>bins[i]) * (xdiff<bins[i+1])] )
       stddiff[i] = np.std( phdiff[(xdiff>bins[i]) * (xdiff<bins[i+1])] )

    """
    plt.figure()
    plt.errorbar((bins[:-1]+bins[1:])/2,meandiff,stddiff)
    plt.xlabel('Pairwise anatomical distance (um)')
    plt.ylabel('Pairwise phase distance')
    """    
    return xdiff,phdiff,meandiff,stddiff,bins
    
#%%
x,y = np.meshgrid(np.arange(-5,6),np.arange(-5,6))
X = np.dstack((x, y))
mu = np.zeros(2)
sigma = np.array([[9,0],[0,9]])
rv = multivariate_normal(mu, sigma)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(randfield)
plt.subplot(2,2,2)
plt.imshow(rv.pdf(X))
plt.subplot(2,2,3)
plt.imshow(convolve_sc(randfield,rv.pdf(X),'same'))

#%%
# circular mean for certain radius
gridx,gridy = np.meshgrid(np.arange(n),np.arange(n))
res = np.zeros(np.shape(randfield))
radius = 20
for i in range(np.shape(randfield)[0]):
    for j in range(np.shape(randfield)[1]):
        dist = np.sqrt((gridx-i)**2 + (gridy-j)**2)
        res[i,j] = circmean(2*np.pi*randfield[dist<radius])
res = res/2/np.pi

xdiff,phdiff,meandiff,stddiff,bins = pairwise_distance(res,gridx,gridy)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(randfield,cmap='twilight_shifted')
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(np.sqrt((gridx-40)**2 + (gridy-30)**2) < radius)
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(res,cmap='twilight_shifted')
plt.colorbar()
plt.subplot(2,2,4)
plt.errorbar((bins[:-1]+bins[1:])/2,meandiff,stddiff)
plt.xlabel('Pairwise anatomical distance')
plt.ylabel('Pairwise phase distance')


#%%
def grid2(X,Y,sc=2./np.sqrt(3), angle=0, offs=np.array([0,0])):
    rec = lambda x: np.where(x>0,x,0)
    if np.size(offs) == 2:
        return rec(np.cos(2*np.pi*sc*np.sin(angle*np.pi/180)*(X-offs[0])+2*np.pi*sc*np.cos(angle*np.pi/180)*(Y-offs[1]))) * rec(np.cos(2*np.pi*sc*np.sin((angle+60)*np.pi/180)*(X-offs[0])+2*np.pi*sc*np.cos((angle+60)*np.pi/180)*(Y-offs[1]))) * rec(np.cos(2*np.pi*sc*np.sin((angle+120)*np.pi/180)*(X-offs[0])+2*np.pi*sc*np.cos((angle+120)*np.pi/180)*(Y-offs[1])))
    else:
        assert len(offs[0]) == len(offs[1]), "ox and oy must have same length"
        return rec(np.cos(2*np.pi*sc*np.sin(angle*np.pi/180)*(X[:,:,None]-(offs[0])[None,None,:])+2*np.pi*sc*np.cos(angle*np.pi/180)*(Y[:,:,None]-(offs[1])[None,None,:]))) * rec(np.cos(2*np.pi*sc*np.sin((angle+60)*np.pi/180)*(X[:,:,None]-(offs[0])[None,None,:])+2*np.pi*sc*np.cos((angle+60)*np.pi/180)*(Y[:,:,None]-(offs[1])[None,None,:]))) * rec(np.cos(2*np.pi*sc*np.sin((angle+120)*np.pi/180)*(X[:,:,None]-(offs[0])[None,None,:])+2*np.pi*sc*np.cos((angle+120)*np.pi/180)*(Y[:,:,None]-(offs[1])[None,None,:])))

def create_randomfield_2D(n,mode='grid'):
    # create 2D random field with grid AC
    #n = 100
    rand_ph_help = 2*np.pi*np.random.rand(n,n) # draw phase of complex number 
    rand_ph_help2 = 2*np.pi*np.random.rand(n,n) # draw phase of complex number 
    randfield1 = np.cos(rand_ph_help)
    randfield1_2 = np.cos(rand_ph_help2)
    randfield2 = np.sin(rand_ph_help)
    randfield2_2 = np.sin(rand_ph_help2)
    x,y = np.meshgrid(np.linspace(0,3,n),np.linspace(0,3,n)) # (3mm)**3 voxel
    
    part = 1
    gridx,gridy = np.meshgrid(np.linspace(-1.5/part,1.5/part,int(n/part)),np.linspace(-1.5/part,1.5/part,int(n/part)))
    X = np.array(list(zip(np.reshape(gridx,(1,-1))[0], np.reshape(gridy,(1,-1))[0])))
    if mode=='gauss':
        mu = np.zeros(2)
        sigma = 0.03
        sig = np.array([[sigma,0],[0,sigma]])**2
        rv = sc.stats.multivariate_normal(mu, sig)
        kernel = np.reshape(rv.pdf(X),(int(n/part),int(n/part)))
    elif mode=='grid':
        grsc = 0.3
        kernel = np.reshape(grid2(X[:,0],X[:,1],sc=1/grsc * 2/np.sqrt(3)),(int(n/part),int(n/part)))
        kernel[np.sqrt(gridx**2 + gridy**2)>1.5*grsc] = 0
        kernel = kernel/np.sum(kernel)/(3/(n-1))**2
    else:
        raise Exception('What are you doing?')
    #plt.figure()
    #plt.imshow(kernel[:,:,5])
    
    res_re = sc.signal.convolve(randfield1,kernel,'same') # use valid instead
    res_re_2 = sc.signal.convolve(randfield1_2,kernel,'same')
    res_im = sc.signal.convolve(randfield2,kernel,'same')
    res_im_2 = sc.signal.convolve(randfield2_2,kernel,'same')
    ph1 = (np.angle(res_re + 1j*res_im) + np.pi)/2/np.pi
    ph2 = (np.angle(res_re_2 + 1j*res_im_2) + np.pi)/2/np.pi
    return ph1,ph2,x,y,kernel

#%%
# filtering real and imaginary part of random complex numbers independently, 3D
def create_randomfield(n,sigma):
    #n = 100
    rand_ph_help = 2*np.pi*np.random.rand(n,n,n) # draw phase of complex number 
    rand_ph_help2 = 2*np.pi*np.random.rand(n,n,n) # draw phase of complex number 
    randfield1 = np.cos(rand_ph_help)
    randfield1_2 = np.cos(rand_ph_help2)
    randfield2 = np.sin(rand_ph_help)
    randfield2_2 = np.sin(rand_ph_help2)
    x,y,z = np.meshgrid(np.linspace(0,3,n),np.linspace(0,3,n),np.linspace(0,3,n)) # (3mm)**3 voxel
    
    part = 1
    gridx,gridy,gridz = np.meshgrid(np.linspace(-1.5/part,1.5/part,int(n/part)),np.linspace(-1.5/part,1.5/part,int(n/part)),np.linspace(-1.5/part,1.5/part,int(n/part)))
    X = np.array(list(zip(np.reshape(gridx,(1,-1))[0], np.reshape(gridy,(1,-1))[0], np.reshape(gridz,(1,-1))[0])))
    mu = np.zeros(3)
    sig = np.array([[sigma,0,0],[0,sigma,0],[0,0,sigma]])**2
    rv = sc.stats.multivariate_normal(mu, sig)
    kernel = np.reshape(rv.pdf(X),(int(n/part),int(n/part),int(n/part)))
    
    #plt.figure()
    #plt.imshow(kernel[:,:,5])
    
    res_re = sc.signal.convolve(randfield1,kernel,'same') # use valid instead
    res_re_2 = sc.signal.convolve(randfield1_2,kernel,'same')
    res_im = sc.signal.convolve(randfield2,kernel,'same')
    res_im_2 = sc.signal.convolve(randfield2_2,kernel,'same')
    ph1 = (np.angle(res_re + 1j*res_im) + np.pi)/2/np.pi
    ph2 = (np.angle(res_re_2 + 1j*res_im_2) + np.pi)/2/np.pi
    return ph1,ph2,x,y,z

def reproduce_gu(ph1,ph2,x,y,z):
    # pick 10^6 random pairs and measure the pairwise distances
    xf, yf, zf = x.flatten(), y.flatten(), z.flatten()
    phf1,phf2 = ph1.flatten(), ph2.flatten()
    reps = 1000000
    ind1 = np.random.randint(0,len(xf),reps)
    ind2 = np.random.randint(0,len(xf),reps)
    xdist = np.sqrt((xf[ind1] - xf[ind2])**2 + (yf[ind1] - yf[ind2])**2 + (zf[ind1] - zf[ind2])**2)
    phfr1,phfr2 = convert_to_rhombus(phf1,phf2)
    phdist1, phdist2 = circdiff(2*np.pi*phfr1[ind1],2*np.pi*phfr1[ind2])/2/np.pi, circdiff(2*np.pi*phfr2[ind1],2*np.pi*phfr2[ind2])/2/np.pi
    phdistr1, phdistr2 = convert_to_rhombus(phdist1,phdist2)
    phdist = np.sqrt(phdistr1**2 + phdistr2**2)
    
    bins = np.linspace(0,0.5,50)
    meandiff = np.zeros(len(bins)-1)
    stddiff = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
       meandiff[i] = np.mean( phdist[(xdist>bins[i]) * (xdist<bins[i+1])] )
       stddiff[i] = np.std( phdist[(xdist>bins[i]) * (xdist<bins[i+1])] )

    return meandiff, stddiff, (bins[:-1]+bins[1:])/2

#%%
# example reproducing Gu et al/Heys et al -- sigma = 30 um
ph1,ph2,x,y,z = create_randomfield(n=200,sigma=0.03)
meandiff, stddiff, bins = reproduce_gu(ph1,ph2,x,y,z)

plt.figure(figsize=(5,12))
plt.subplot(3,1,1)
plt.imshow(ph1[:,:,5],origin='lower',cmap='twilight_shifted',vmin=0, vmax=1)
plt.xticks(np.linspace(0,200,4),['0','1','2','3'])
plt.yticks(np.linspace(0,200,4),['0','1','2','3'])
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar()
plt.title('Grid phase (X)')
plt.subplot(3,1,2)
plt.hist(ph1.flatten(),density=True)
plt.xlabel('Grid phase (X)')
plt.subplot(3,1,3)
plt.errorbar(bins * 1000,meandiff,stddiff)
plt.xlabel('Pairwise anatomical distance (um)')
plt.ylabel('Pairwise phase distance (2D)')

#%%
# test different sigmas
plt.figure()
plt.xlabel('Pairwise anatomical distance (um)')
plt.ylabel('Pairwise phase distance')
nplot = 5
sigmas = np.logspace(-2,0,nplot)
for i,sigma in enumerate(sigmas):
    print(i)
    print(sigma)
    ph1,ph2,x,y,z = create_randomfield(n=100,sigma=sigma)
    meandiff, stddiff, bins = reproduce_gu(ph1,ph2,x,y,z)
    plt.errorbar(bins * 1000,meandiff,stddiff,ecolor=str(i/n))
plt.legend([str(np.round(s,2)) for s in sigmas])

#%%
# measure von mises' kappa for different kernel sigmas
ns = 100
sigmas = np.logspace(-2,1,ns)
kappax = np.zeros(ns)
kappay = np.zeros(ns)
for i,sigma in enumerate(sigmas):
    print(i)
    print(sigma)
    ph1,ph2,x,y,z = create_randomfield(n=100,sigma=sigma)
    kappax[i],mux,fs = sc.stats.vonmises.fit(2*np.pi*ph1.flatten(), fscale=1)
    kappay[i],muy,fs = sc.stats.vonmises.fit(2*np.pi*ph2.flatten(), fscale=1)

plt.figure()
plt.plot(sigmas,kappax,'bo',sigmas,kappay,'go')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Kernel sigma (mm)')
plt.ylabel('Von Mises kappa')
plt.legend(['kappa_x','kappa_y'],loc=2)
plt.ylim([1e-4,1e1])
plt.plot([0.03,0.03],[1e-4,1e1],'k--')
plt.plot([3,3],[1e-4,1e1],'k--')

#%%
# plot example for sigma=2mm
ph1,ph2,x,y,z = create_randomfield(n=200,sigma=2)
meandiff, stddiff, bins = reproduce_gu(ph1,ph2,x,y,z)

plt.figure(figsize=(4,12))
plt.subplot(3,1,1)
plt.imshow(ph1[:,176,:],origin='lower',cmap='twilight_shifted',vmin=0, vmax=1)
plt.xticks(np.linspace(0,200,4),['0','1','2','3'])
plt.yticks(np.linspace(0,200,4),['0','1','2','3'])
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar()
plt.title('Grid phase (X)')
plt.subplot(3,1,2)
plt.hist(ph1.flatten(),density=True)
plt.xlabel('Grid phase (X)')
plt.subplot(3,1,3)
plt.errorbar(bins * 1000,meandiff,stddiff)
plt.xlabel('Pairwise anatomical distance (um)')
plt.ylabel('Pairwise phase distance (2D)')

#%%
# 2D random field with grid AC
ph1,ph2,x,y,kernel = create_randomfield_2D(n=200,mode='grid')
ph1c,ph2c,xc,yc,kernelc = create_randomfield_2D(n=200,mode='gauss')
meandiff, stddiff, bins_1 = reproduce_gu(ph1,ph2,x,y,np.zeros(np.shape(x)))
meandiffc, stddiffc, binsc = reproduce_gu(ph1c,ph2c,xc,yc,np.zeros(np.shape(xc)))

from scipy.stats import vonmises,ttest_rel,mannwhitneyu

plt.figure(figsize=(16,8))
plt.subplot(2,4,1)
plt.imshow(ph1,origin='lower',cmap='twilight_shifted',vmin=0, vmax=1)
plt.xticks(np.linspace(0,200,4),['0','1','2','3'])
plt.yticks(np.linspace(0,200,4),['0','1','2','3'])
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar()
plt.title('Grid phase (X)')
plt.subplot(2,4,5)
plt.imshow(ph1c,origin='lower',cmap='twilight_shifted',vmin=0, vmax=1)
plt.xticks(np.linspace(0,200,4),['0','1','2','3'])
plt.yticks(np.linspace(0,200,4),['0','1','2','3'])
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar()
plt.subplot(2,4,2)
plt.imshow(kernel,origin='lower')
plt.xticks(np.linspace(0,200,4),['0','1','2','3'])
plt.yticks(np.linspace(0,200,4),['0','1','2','3'])
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar()
plt.subplot(2,4,6)
plt.imshow(kernelc,origin='lower')
plt.xticks(np.linspace(0,200,4),['0','1','2','3'])
plt.yticks(np.linspace(0,200,4),['0','1','2','3'])
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar()
plt.subplot(2,4,3)
plt.hist(ph1.flatten(),density=True)
plt.title('kappa = '+str(np.round(vonmises.fit(2*np.pi*ph1, fscale=1)[0],3)))
plt.xlabel('Grid phase (X)')
plt.subplot(2,4,7)
plt.hist(ph1c.flatten(),density=True)
plt.title('kappa = '+ str(np.round(vonmises.fit(2*np.pi*ph1c, fscale=1)[0],3)))
plt.xlabel('Grid phase (X)')
plt.subplot(2,4,4)
plt.errorbar(bins_1 * 1000,meandiff,stddiff)
plt.xlabel('Pairwise anatomical distance (um)')
plt.ylabel('Pairwise phase distance (2D)')
plt.subplot(2,4,8)
plt.errorbar(binsc * 1000,meandiffc,stddiffc)
plt.xlabel('Pairwise anatomical distance (um)')
plt.ylabel('Pairwise phase distance (2D)')
plt.tight_layout()

#%% save stuff
data = np.array([binsc,meandiffc,stddiffc,bins_1,meandiff,stddiff]).T
with open('binned_phasedistance.txt', 'a') as f:
    for x in data:
        f.write(str(x)+'\n')

with open('kernel_gauss.txt', 'a') as f:
    for x in kernelc:
        f.write(str(x)+'\n')

with open('kernel_grid.txt', 'a') as f:
    for x in kernel:
        f.write(str(x)+'\n')

with open('ph1_grid.txt', 'a') as f:
    for x in ph1:
        f.write(str(x)+'\n')

with open('ph2_grid.txt', 'a') as f:
    for x in ph2:
        f.write(str(x)+'\n')

with open('ph1c_gauss.txt', 'a') as f:
    for x in ph1c:
        f.write(str(x)+'\n')

with open('ph2c_gauss.txt', 'a') as f:
    for x in ph2c:
        f.write(str(x)+'\n')

#%%
rep = 100
ii=0
kx_grid = np.zeros(rep)
ky_grid = np.zeros(rep)
kx_gauss = np.zeros(rep)
ky_gauss = np.zeros(rep)
while ii<rep:
    print(ii)
    ph1,ph2,x,y,kernel = create_randomfield_2D(n=200,mode='grid')
    ph1c,ph2c,xc,yc,kernelc = create_randomfield_2D(n=200,mode='gauss')
    kx_grid[ii],ky_grid[ii] = vonmises.fit(2*np.pi*ph1, fscale=1)[0], vonmises.fit(2*np.pi*ph2, fscale=1)[0]
    kx_gauss[ii],ky_gauss[ii] = vonmises.fit(2*np.pi*ph1c, fscale=1)[0], vonmises.fit(2*np.pi*ph2c, fscale=1)[0]
    ii+=1

plt.figure()
plt.violinplot([kx_gauss,ky_gauss,kx_grid,ky_grid],showmedians=True)
plt.xticks([1,2,3,4],['kappa_x Gauss', 'kappa_y Gauss','kappa_x grid','kappa_y grid'])
plt.title('Random field simulations for the clustering hypotheses')

mannwhitneyu(kx_gauss,kx_grid)