import numpy as np
import occultquad

def ModLC(timeIn, RpRs, bsq, tR, t0, u1, u2):
    z0   = np.sqrt(bsq + ((timeIn-t0) / tR)**2)
    nz   = len(timeIn)
    muo1 = np.zeros((nz))
    mu0  = np.zeros((nz))
    _    = occultquad.occultquad(z0,u1,u2,RpRs,muo1,mu0,nz)
    return muo1


