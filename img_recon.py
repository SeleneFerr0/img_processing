from numpy import linspace, zeros, cumsum, ones, where,meshgrid, shape
from numpy.random import rand, randn
from skimage.color import rgb2gray
from skimage.io import imshow, imread
from skimage.exposure import equalize_hist as histeq
from skimage.exposure import cumulative_distribution as cdf
from matplotlib.colors import ListedColormap as colormap
import matplotlib.pyplot as plt
from  matplotlib.pyplot import  xlabel, ylabel,title
from scipy.ndimage.filters import median_filter as medfil2d
from skimage.filters.rank import maximum
from skimage.filters.rank import gradient as grad
from skimage.filters import threshold_otsu as greythres
import numpy as np
import skimage.data as Images
from scipy.signal import convolve2d as filter2D
from numpy.fft import fft, fft2, ifft, ifft2, fftshift
from skimage.transform import rotate, radon, iradon, hough_line_peaks,hough_line
from skimage.morphology import disk,square,rectangle
from skimage.restoration import wiener
from skimage.restoration import deconvolution
from skimage.morphology import binary_erosion as imerode
from skimage.morphology import binary_dilation as imdilate
from skimage.morphology import binary_opening as imopen
from skimage.morphology import binary_closing  as imclose
from skimage.morphology import erosion as gs_imerode
from skimage.morphology import dilation as gs_imdilate
from skimage.morphology import opening as gs_imopen
from skimage.morphology import closing  as gs_imclose
from skimage.morphology import black_tophat  as tophat
from skimage.transform import warp, AffineTransform,ProjectiveTransform
from skimage import data, color
from scipy.interpolate import griddata
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import canny, peak_local_max
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from scipy.io import loadmat, savemat
from numba import jit
from pylab import imshow, plot, subplot, subplots, show
#import tensorflow as tf
from PIL import Image

def imresize(I,xdim,ydim):
    imsze=[xdim,ydim]
    return(np.array(Image.fromarray(I).resize(imsze)))
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
def sigmod(x,derivative=False):
    sigm=1/(1 + np.exp(-x))
    if derivative:
        return sigm * (1 - sigm)
    return sigm

def image_subplot(img,nrow=1,ncol=1, map1='gray', **kwargs):
    fig,ax = plt.subplots(nrow,ncol,figsize=(16,16))
    ax.imshow(img,cmap=map1,**kwargs)
    ax.axis('off')
    return(fig, ax)

def bitget(I,Ival):
    return (I>>Ival) % 2

def hough(img_bin, theta_res=1, rho_res=1):
    """
     Computes the Hough transform of an image
    """
    nR,nC = img_bin.shape
    theta = np.linspace(-90.0, 0.0, np.ceil(90.0/theta_res) + 1.0)
    theta = np.concatenate((theta, -theta[len(theta)-2::-1]))

    D = np.sqrt((nR - 1)**2 + (nC - 1)**2)
    q = np.ceil(D/rho_res)
    nrho = 2*q + 1
    rho = np.linspace(-q*rho_res, q*rho_res, nrho)
    H = np.zeros((len(rho), len(theta)))
    for rowIdx in range(nR):
        for colIdx in range(nC):
            if img_bin[rowIdx, colIdx]:
                for thIdx in range(len(theta)):
                    rhoVal = colIdx*np.cos(theta[thIdx]*np.pi/180.0) + \
                             rowIdx*np.sin(theta[thIdx]*np.pi/180)
                    rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
                    H[rhoIdx[0], thIdx] += 1
    return H, theta,rho



def myImshow(I0,cmp="gray"):
    imshow(I0,aspect="auto",cmap=cmp)
def im2bw(Ig,level):
    S=np.copy(Ig)
    S[Ig > level] = 1
    S[Ig <= level] = 0
    return(S)
def dtfuv(m,n):
    """
    Computes the frequency matrices, used to construct frequency domain filters

    """
    m = m  # double size to detal with wrap round effects
    n = n
    u=linspace(0,m-1,m)
    v=linspace(0,n-1,n)

    idx = where(u > m/2)
    u[idx] = u[idx]-m

    idy = where(v > n/2)
    v[idy] = v[idy]-n

    V,U = meshgrid(v,u)
    return (V,U)

def fftfilt(f,H):
    """
    This function performs fourier domain filtering. I also to zero padding to prevent wrap around effects
    output filtered imaged
    f = input image
    H =  filter coeffienets
    """
    f = f.astype("double")
    I = fft2(f,(shape(H)[0],shape(H)[1]))
    I2 = ifft2(H*I)
    return I2[:shape(f)[0],:shape(f)[1]].real


def Phi(x):
    n=len(x)
    output=np.double(np.logical_and(x>=0,x<=1))
    return (np.reshape(output,(n,1)))

def Psi(x):
    ph1=np.zeros((len(x),1))
    n=len(x)
    for ii in range(n):
        if(x[ii] >=0.5 and x[ii]<=1):
            ph1[ii]=-1
        elif (x[ii] >= 0. and x[ii] <=0.5):
            ph1[ii]=1
        else:
                    ph1[ii]=0
    return(ph1)

def phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None):
        """
         phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)

        Create a Shepp-Logan or modified Shepp-Logan phantom.

        A phantom is a known object (either real or purely mathematical)
        that is used for testing image reconstruction algorithms.  The
        Shepp-Logan phantom is a popular mathematical model of a cranial
        slice, made up of a set of ellipses.  This allows rigorous
        testing of computed tomography (CT) algorithms as it can be
        analytically transformed with the radon transform (see the
        function `radon').

        Inputs
        ------
        n : The edge length of the square image to be produced.

        p_type : The type of phantom to produce. Either
          "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
          if `ellipses' is also specified.

        ellipses : Custom set of ellipses to use.  These should be in
          the form
                [[I, a, b, x0, y0, phi],
                 [I, a, b, x0, y0, phi],
                 ...]
          where each row defines an ellipse.
          I : Additive intensity of the ellipse.
          a : Length of the major axis.
          b : Length of the minor axis.
          x0 : Horizontal offset of the centre of the ellipse.
          y0 : Vertical offset of the centre of the ellipse.
          phi : Counterclockwise rotation of the ellipse in degrees,
                measured as the angle between the horizontal axis and
                the ellipse major axis.
          The image bounding box in the algorithm is [-1, -1], [1, 1],
          so the values of a, b, x0, y0 should all be specified with
          respect to this box.

        Output
        ------
        P : A phantom image.

        Usage example
        -------------
          import matplotlib.pyplot as pl
          P = phantom ()
          pl.imshow (P)

        References
        ----------
        Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
        from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
        Feb. 1974, p. 232.

        Toft, P.; "The Radon Transform - Theory and Implementation",
        Ph.D. thesis, Department of Mathematical Modelling, Technical
        University of Denmark, June 1996.

        """

        if (ellipses is None):
                ellipses = _select_phantom (p_type)
        elif (np.size (ellipses, 1) != 6):
                raise AssertionError ("Wrong number of columns in user phantom")

        # Blank image
        p = np.zeros ((n, n))

        # Create the pixel grid
        ygrid, xgrid = np.mgrid[-1:1:(1j*n), -1:1:(1j*n)]

        for ellip in ellipses:
                I   = ellip [0]
                a2  = ellip [1]**2
                b2  = ellip [2]**2
                x0  = ellip [3]
                y0  = ellip [4]
                phi = ellip [5] * np.pi / 180  # Rotation angle in radians

                # Create the offset x and y values for the grid
                x = xgrid - x0
                y = ygrid - y0

                cos_p = np.cos (phi)
                sin_p = np.sin (phi)

                # Find the pixels within the ellipse
                locs = (((x * cos_p + y * sin_p)**2) / a2
              + ((y * cos_p - x * sin_p)**2) / b2) <= 1

                # Add the ellipse intensity to those pixels
                p [locs] += I

        return p


def _select_phantom (name):
        if (name.lower () == 'shepp-logan'):
                e = _shepp_logan ()
        elif (name.lower () == 'modified shepp-logan'):
                e = _mod_shepp_logan ()
        else:
                raise ValueError ("Unknown phantom type: %s" % name)

        return e


def _shepp_logan ():
        #  Standard head phantom, taken from Shepp & Logan
        return [[   2,   .69,   .92,    0,      0,   0],
                [-.98, .6624, .8740,    0, -.0184,   0],
                [-.02, .1100, .3100,  .22,      0, -18],
                [-.02, .1600, .4100, -.22,      0,  18],
                [ .01, .2100, .2500,    0,    .35,   0],
                [ .01, .0460, .0460,    0,     .1,   0],
                [ .02, .0460, .0460,    0,    -.1,   0],
                [ .01, .0460, .0230, -.08,  -.605,   0],
                [ .01, .0230, .0230,    0,  -.606,   0],
                [ .01, .0230, .0460,  .06,  -.605,   0]]

def _mod_shepp_logan ():
        #  Modified version of Shepp & Logan's head phantom,
        #  adjusted to improve contrast.  Taken from Toft.
        return [[   1,   .69,   .92,    0,      0,   0],
                [-.80, .6624, .8740,    0, -.0184,   0],
                [-.20, .1100, .3100,  .22,      0, -18],
                [-.20, .1600, .4100, -.22,      0,  18],
                [ .10, .2100, .2500,    0,    .35,   0],
                [ .10, .0460, .0460,    0,     .1,   0],
                [ .10, .0460, .0460,    0,    -.1,   0],
                [ .10, .0460, .0230, -.08,  -.605,   0],
                [ .10, .0230, .0230,    0,  -.606,   0],
                [ .10, .0230, .0460,  .06,  -.605,   0]]


def convMx(F,I):
    from scipy.linalg import toeplitz
    I_row_num, I_col_num= I.shape
    F_row_num, F_col_num = F.shape
    output_row_num = I_row_num + F_row_num - 1
    output_col_num = I_col_num + F_col_num - 1
    
    
    print(output_row_num,output_col_num)
    F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0),(0, output_col_num - F_col_num)), 'constant', constant_values=0)
    toeplitz_list = []
    for i in range(F_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
        c = F_zero_padded[i, :] # i th row of the F 
        r = np.r_[c[0], np.zeros(I_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                        # the result is wrong
        toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
        toeplitz_list.append(toeplitz_m)

    # Create blocked toeplitz matrix
    c = range(1, F_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(I_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c, r)
    
    ## creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)
    
    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]
    return (doubly_blocked) 


def matrix_to_vector(input):
    input_h, input_w = input.shape
    output_vector = np.zeros(input_h*input_w, dtype=input.dtype)
    # flip the input matrix up-down because last row should go first
    input = np.flipud(input) 
    for i,row in enumerate(input):
        st = i*input_w
        nd = st + input_w
        output_vector[st:nd] = row
        
    return output_vector


def vector_to_matrix(input, output_shape):
    output_h, output_w = output_shape
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range(output_h):
        st = i*output_w
        nd = st + output_w
        output[i, :] = input[st:nd]
    # flip the output matrix up-down to get correct result
    output=np.flipud(output)
    return output

def hilbert(x):
    sh = np.shape(x);
    row = sh[0]; col = sh[1];
    h = np.zeros((1,row));# + 1j*zeros((1,row));
    result = np.zeros(np.shape(x)) + 1j*np.zeros(np.shape(x))
    if (row%2 == 0):
        h[0,0] = 1;
        h[0,int(row/2)] = 1;
        h[0,1:int(row/2)] = 2;
    else:
        h[0,0] = 1;
        h[0,1:(row+1)/2] = 2
    for i in range(0,col):
        sig = x[:,i];
        result[:,i] = ifft(fft(sig)*h);
   # Returns output like matlab's hilbert real component is the orginal data and the imaginary component is the the 90 degrees phase flip data
    return (x+i*result)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

# Delay computation for simple phase_array
# All computations is performed in polar coordinates

@jit(nopython=True, cache=True)
def DAS_phaseArray(rf0,info,xx,zz,W,theta2=0.0):
    samp = np.int64(info['Samples'][0])
    alines = np.int64(info['Alines'][0])
    Tx = np.int64(info['Tx'][0])
    rf = np.zeros((samp,alines,Tx),dtype=np.float64)

    startRxTime = (2*np.float64(info['startDepth'][0]))/np.float64(info['c'][0])
    helm = (Tx-1)/2
    ele_dist = np.arange(-helm,helm+1)*np.float64(info['pitch'][0])                      # Steering angle of elements
    theta2 = theta2 * np.pi/180

    xx =xx * np.pi/(180)                # Angles scanned
    for ii in range (samp):
        for jj in range (alines):
            theta1=xx[ii,jj]  # Grid point x-coordinate
            rad1=zz[ii,jj]   # radial distance

            for k in range(Tx):
                z1 = rad1*np.cos(theta1+theta2)
                x1 = rad1*np.sin(theta1+theta2)
                tdx = z1
                rdx = np.sqrt((x1-ele_dist[k])**2 + z1**2)
                dist = (tdx+rdx)/np.float64(info['c'][0])
                sampNum = (dist-startRxTime)*np.float64(info['fs'][0])
           
                sampNumi = int(np.floor(sampNum)) # whole samples
            
                frac = sampNum - sampNumi    # fractional samples 
                tmp = 0.0
                if (sampNumi >=0 and sampNumi < samp - 1):
                    tmp = ((1-frac)*rf0[sampNumi,k]) + (frac * rf0[sampNumi+1, k])

                rf[ii,jj,k]=tmp  # delayed waveforms

    rf = rf * W  # apply apodization

    return (np.sum(rf,axis=2))

# Delay and sum beamformer for linear array transducer
@jit(nopython=True)
def DAS_linearArray(rf0,samp,alines,Tx,c,startDepth,fs,pitch,xx,zz,nn,W):
  
    rf = np.zeros((samp,alines,Tx),dtype=np.float64)

    startRxTime = (2*startDepth)/c
    helm = (Tx-1)/2
    ele_dist = np.arange(-helm,helm+1)*pitch                      # Steering angle of elements
            # Angles scanned
    for ii in range (samp):
        for jj in range (alines):
            x1=xx[ii,jj]   # Grid point x-coordinate
            z1=zz[ii,jj]   # radial distance

            for k in range(Tx):
                tdx = z1
                rdx = np.sqrt((x1-ele_dist[k])**2 + z1**2)
                dist = (tdx+rdx)/c
                sampNum = (dist-startRxTime)*fs
           
                sampNumi = int(np.floor(sampNum)) # whole samples
            
                frac = sampNum - sampNumi    # fractional samples 
                tmp = 0.0
                if (sampNumi >=0 and sampNumi < nn-1):
                    tmp = ((1-frac)*rf0[sampNumi,k]) + (frac * rf0[sampNumi+1, k])

                rf[ii,jj,k]=tmp  # delayed waveforms

    rf = rf * W  # apply apodization

    return (np.sum(rf,axis=2))
