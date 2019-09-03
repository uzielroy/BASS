#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:45:03 2018

@author: uzielr
"""


"""
************************************************************************************************************************************************************

                                                                            Imports

************************************************************************************************************************************************************
"""

from BASS import *
import torch
import os
import sys

import numpy as np

import cv2
import warnings


warnings.filterwarnings("ignore")


from IPython.terminal.embed import InteractiveShellEmbed
ip = InteractiveShellEmbed( banner1 = 'Dropping into IPython',
                                     exit_msg = 'Leaving IPython, back to program.')



"""
************************************************************************************************************************************************************

                                                                        Help Funtions

************************************************************************************************************************************************************
"""



def Calc_H(Theta):
    H[:,:,0]=np.cos(Theta)
    H[:,:,1]=np.sin(Theta)
    return H


def Calc_Y(X,H):
    #TODO: Matrix Mul
    for i in range (0,N):
        Y[:,i]=H[i]@X[i,:]
    return Y

def Calc_ProjPoints(Y,H):
    ProjPoints=np.zeros((N,D))
    for i in range (0,N):
        ProjPoints[i]=Y[:,i]@H[i]
    return ProjPoints



def hex2rgb(value):
    value =value.lstrip('#')
    r,g,b= tuple(int(value[i:i+2], 16) for i in (0, 2 ,4))
    return [r,g,b]

def blockshaped(arr, nrows, ncols):


    """Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.

    **Parameters**:
     - arr - 2d matrix to split

     - nrows - Size of row after the split

     - ncols - Size of col after the split

    **Returns**:
     - arr - Array after the split

    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):

    """Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    Use after :meth:`AngleImpl.blockshaped`.
    Create_DataMatrix
    **Parameters**:
     - arr - 2d matrix after split

     - nrows - Size of row before the split

     - ncols - Size of col before the split

    **Returns**:
     - arr - Array before the split

    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))



def softmax(x,axis):

    """Compute softmax values for each sets of scores in x.
    **Parameters**:
      - X - array of point.
      - axis - axis to sum over

    **Returns**:
      - arr -  Exp of the data after softmax

    """
    e_x = np.exp(x - np.max(x,axis=0)[np.newaxis])
    return e_x / e_x.sum(axis)[np.newaxis]


def softmaxTF(x, axis,sum):
    """Compute softmax values for each sets of scores in x.
    **Parameters**:
      - X - array of point.
      - axis - axis to sum over

    **Returns**:
      - arr -  Exp of the data after softmax

    """
    sum.zero_()
    xmax=x.max(dim=axis)[0].unsqueeze(axis)
    x.sub_(xmax)
    x.exp_()
    if(Global.neig_num==5):
        x.div_(sum.add_(x[:,0]).add_(x[:,1]).add_(x[:,2]).add_(x[:,3]).add_(x[:,4]).unsqueeze(axis))
    if(Global.neig_num==4):
        x.div_(sum.add_(x[:, 0]).add_(x[:, 1]).add_(x[:, 2]).add_(x[:, 3]).unsqueeze(axis))
   # e_x = torch.exp(torch.sub(x,x.max(dim=axis)[0].unsqueeze(axis)))
   # return torch.div(e_x,(sum.add_(e_x[:,0]).add_(e_x[:,1]).add_(e_x[:,2]).add_(e_x[:,3]).add_(e_x[:,4])).unsqueeze(axis))


def Create_DataMatrix(figure):
    figure=cv2.cvtColor(figure, cv2.COLOR_RGB2LAB).astype("float32")
    x,y=np.mgrid[:figure.shape[0],:figure.shape[1]]
    L=figure[:,:,0].ravel()
    A=figure[:,:,1].ravel()
    B=figure[:,:,2].ravel()

    data=np.array((x.ravel(),y.ravel(),L,A,B))
    return data.T

def apply(func, M):
    tList = [func(m.cuda(async=True)) for m in torch.unbind(M, dim=0) ]
    res = torch.stack(tList, dim=0)

    return res


def makeColorwheel():

	#  color encoding scheme

	#   adapted from the color circle idea described at
	#   http://members.shaw.ca/quadibloc/other/colint.htm

	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3]) # r g b

	col = 0
	#RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
	col += RY

	#YG
	colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
	colorwheel[col:YG+col, 1] = 255;
	col += YG;

	#GC
	colorwheel[col:GC+col, 1]= 255
	colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
	col += GC;

	#CB
	colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
	colorwheel[col:CB+col, 2] = 255
	col += CB;

	#BM
	colorwheel[col:BM+col, 2]= 255
	colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
	col += BM;

	#MR
	colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
	colorwheel[col:MR+col, 0] = 255
	return 	colorwheel

def computeColor(u, v):

	colorwheel = makeColorwheel();
	nan_u = np.isnan(u)
	nan_v = np.isnan(v)
	nan_u = np.where(nan_u)
	nan_v = np.where(nan_v)

	u[nan_u] = 0
	u[nan_v] = 0
	v[nan_u] = 0
	v[nan_v] = 0

	ncols = colorwheel.shape[0]
	radius = np.sqrt(u**2 + v**2)
	a = np.arctan2(-v, -u) / np.pi
	fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
	k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
	k1 = k0+1;
	k1[k1 == ncols] = 0
	f = fk - k0

	img = np.empty([k1.shape[0], k1.shape[1],3])
	ncolors = colorwheel.shape[1]
	for i in range(ncolors):
		tmp = colorwheel[:,i]
		col0 = tmp[k0]/255
		col1 = tmp[k1]/255
		col = (1-f)*col0 + f*col1
		idx = radius <= 1
		col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius
		col[~idx] *= 0.75 # out of range
		img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

	return img.astype(np.uint8)


def computeImg(flow):

    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10

    u = flow[: , : , 0]
    v = flow[: , : , 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999

    maxrad = -1
    #fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0
    v[greater_v] = 0

    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])

    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v))
    maxrad = max([maxrad, np.amax(rad)])
    print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

    u = u/(maxrad+eps)
    v = v/(maxrad+eps)
    img = computeColor(u, v)
    return img

def ReadfloFile(file):
    f=open(file, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32, count=2 * w * h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (h, w, 2))
    return data2D
