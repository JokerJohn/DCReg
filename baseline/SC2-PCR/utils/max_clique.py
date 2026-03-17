#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Code borrowed from https://github.com/ryanrossi/pmc/blob/master/pmc.py
Find maxumum clique in given dimacs-format graph
based on:
http://www.m-hikari.com/ams/ams-2014/ams-1-4-2014/mamatAMS1-4-2014-3.pdf
'''
import os
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes

def pmc(ei,ej,nnodes,nnedges): #ei, ej is edge list whose index starts from 0
    degrees = np.zeros(nnodes,dtype = np.int32)
    new_ei = []
    new_ej = []
    for i in range(nnedges):
        degrees[ei[i]] += 1
        if ej[i] <= ei[i] + 1:
            new_ei.append(ei[i])
            new_ej.append(ej[i])
    maxd = max(degrees)
    offset = 0
    new_ei = np.array(new_ei,dtype = np.int32)
    new_ej = np.array(new_ej,dtype = np.int32)
    outsize = maxd
    output = np.zeros(maxd,dtype = np.int32)
    lib = ctypes.cdll.LoadLibrary(os.path.abspath("utils/libpmc.so"))
    fun = lib.max_clique
    #call C function
    fun.restype = np.int32
    fun.argtypes = [ctypes.c_int32,ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),ctypes.c_int32,
                  ctypes.c_int32,ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS")]
    clique_size = fun(len(new_ei),new_ei,new_ej,offset,outsize,output)
    max_clique = np.empty(clique_size,dtype = np.int32)
    max_clique[:]=[output[i] for i in range(clique_size)]

    return max_clique
