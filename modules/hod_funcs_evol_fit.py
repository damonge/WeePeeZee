import numpy as np
import pyccl as ccl
# import logging
# logging.basicConfig(level=logging.INFO)

class HODParams(object):

    def __init__(self, hodpars, islogm0=False, islogm1=False, verbose=False):

        # self.log = logging.getLogger('HODParams')
        # self.log.setLevel(logging.INFO)
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(levelname)s: %(message)s')
        # ch.setFormatter(formatter)
        # self.log.addHandler(ch)
        # self.log.propagate = False

        self.params = hodpars
        self.islogm0 = islogm0
        self.islogm1 = islogm1
        if verbose:
            print('Parameters updated: hodpars = {}.'.format(hodpars))

        return

    def lmminf(self, z) :
        #Returns log10(M_min)
        if 'zfid' in self.params:
            lmmin = self.params['lmmin'] + self.params['lmminp']*(1./(1. + z)-(1./(1. + self.params['zfid'])))
        elif 'lmmin' in self.params:
            lmmin = self.params['lmminp']*(1. - 1./(1. + z)) + self.params['lmmin']
        else:
            lmmin = self.params['lmmin_0']/(1. + z) + self.params['lmmin_1']
        return lmmin

    def sigmf(self, z):
        sigm = self.params['sigm_0']*(1. + z)**self.params['sigm_1']
        return sigm

    def m0f(self, z) :
        # Returns M_0
        if 'zfid' in self.params:

            m0 = self.params['m0'] + self.params['m0p']*(1/(1. + z)-(1./(1. + self.params['zfid'])))
        elif 'm0' in self.params:
            m0 = self.params['m0p']*(1. - 1./(1. + z)) + self.params['m0']
        else:
            m0 = self.params['m0_0']*z + self.params['m0_1']
        if self.islogm0:
            m0 = 10**m0
        return m0

    def m1f(self, z) :
        #Returns M_1
        if 'zfid' in self.params:
            m1 = self.params['m1'] + self.params['m1p']*(1/(1. + z)-(1./(1. + self.params['zfid'])))
        elif 'm1' in self.params:
            m1 = self.params['m1p']*(1. - 1./(1. + z)) + self.params['m1']
        else:
            m1 = self.params['m1_0']/(1. +z) + self.params['m1_1']
        if self.islogm1:
            m1 = 10**m1
        return m1

    def alphaf(self, z) :
        #Returns alpha
        alpha = self.params['alpha_0']*(1. + z)**self.params['alpha_1']
        return alpha

    def fcf(self, z) :
        #Returns f_central
        fc = self.params['fc_0']*(1. + z)**self.params['fc_1']
        return fc
