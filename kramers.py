import numpy as np
"""
Usage
-----

MacLaurin KKT:
mcdfreal = kk_maclaurin(Eax, epsilon.imag)
mcdfimag = rkk_maclaurin(Eax, epsilon.real)
mclfreal = - kk_maclaurin(Eax, np.imag(-1./epsilon)) + 2
mclfimag = - rkk_maclaurin(Eax, np.real(1./epsilon))

Fourier KKT
fcdfreal = 2 - kk_fourier(Eax, epsilon.imag)
fcdfimag = - rkk_fourier(Eax, epsilon.real)
flfreal = kk_fourier(Eax, np.imag(-1./epsilon))
flfimag = rkk_fourier(Eax, np.real(1./epsilon))
"""
def kk_maclaurin(e, im):
    re = np.zeros_like(im)
    presum = 2*(e[1]-e[0])/np.pi
    ee = e*e
    for i in range(len(im)):
        mask = e != e[i]
        re[i] = np.sum(e[mask]*im[mask]/(ee[mask]-e[i]*e[i]))
    return presum * re + 1 #+1 is needed only for epsilon1 calculation for xx, yy and zz projections

def rkk_maclaurin(e, re):
    im = np.zeros_like(re)
    presum = 2*(e[1]-e[0])/np.pi
    ee = e*e
    for i in range(len(im)):
        mask = e != e[i]
        im[i] = np.sum(e[i]*(re[mask]-1)/(e[i]*e[i]-ee[mask]))
    return presum * im

def kk_fourier(e, im):
    re = np.zeros_like(im)
    esize = len(e)
    dsize = 2 * esize
    q = - 2 * np.fft.fft(im, dsize, -1).imag / dsize
    #q = - np.fft.fft(np.pad(im, (0, esize), mode='reflect', reflect_type='odd')).imag / dsize
    q[:esize] *= -1.
    q = np.fft.fft(q, axis=-1)
    return q[:esize].real + 1.

def rkk_fourier(e, re):
    im = np.zeros_like(re)
    esize = len(e)
    dsize = 2 * esize
    #q = - 2 * np.fft.fft(re, dsize, -1).real / dsize
    q = - np.fft.fft(np.pad(re, (0, esize), mode='reflect')).real / dsize
    q[:esize] *= -1.
    q = np.fft.fft(q, axis=-1)
    return q[:esize].imag
