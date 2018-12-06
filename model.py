'''
This file contains scripts to perform model-based analysis of the relativistic
Kramers-Kronig method. The model consists in a dielectric function based on the
Tauc-Lorenz model, calculated for several band gaps. This model dataset is then
repeated for several thicknesses.
'''
import hyperspy.api as hs
import numpy as np

# own dependencies
from dielectric import ModifiedCDF
from eels import  ModifiedEELS

def fourier_exp_convolution(ssd, zlp):
    '''
    Using Poisson distribution statistics to simulate the effect of plural
    scattering; starting from an input single-scattering distribution (SSD) and
    zero-loss peak (ZLP) models, we use the exponential formula from Egerton
    (see eq. 4.8, pp 233). This is implemented using FFTs as the product of the
    exponential of the SSD and the ZLP. Proper care should be taken of the shift
    of the energy axis and the normalization by the ZLP intensity. The ratio
    between the SSD and ZLP intensities sets the simulated mean-free-path.

    Note that because FEC uses FFT, care should be taken to ensure the input
    SSD and ZLP decay smoothly to zero at both ends of the energy-loss axis.
    The algorithm also assumes both signals start at the same energy-loss value
    and have the same scale, but not necessarily the same number of channels.

    Parameters
    ----------
    ssd : signal1D
     A signal containing the SSD, for instance a spectrum obtained from a
     dielectric function after calling "get_relativistic_spectrum".
    zlp : signal1D
     A signal containing the ZLP, for instance a model gaussian function with
     fixed width and intensity.

    Returns
    -------
    eels : signal1D
     A signal representing the plural scattering spectrum corresponding to the
     input SSD and ZLP. Fourier-log deconvolution can be applied to this signal
     to obtain an estimate of the SSD;
     >>> ssd_estimate = eels.fourier_log_deconvolution(zlp)

    References
    ----------
    R.F. Egerton "EELS in the electron microscope" 2011, Springer.
    '''
    eels = ssd.deepcopy()
    axis = ssd.axes_manager[-1]
    #ones_like_navi = np.ones(ssd.axes_manager._navigation_shape_in_array)

    tsize = (2*axis.size)

    # preload time-shift
    tdata = np.exp(-2j*np.pi*axis.offset/axis.scale*np.fft.fftfreq(tsize))#*ones_like_navi[...,None])

    # preload ZLP integral
    I0 = zlp.integrate1D(-1).data**-1#[..., None]
    I0 = I0[..., None] if ssd.axes_manager.navigation_dimension > 0 else I0

    # The FFTs are done with 2 times energy-loss axis size
    kwfft = {'n':tsize, 'axis':axis.index_in_array}
    eels.data = np.exp(I0*tdata*np.fft.fft(ssd.data, **kwfft))
    eels.data *= tdata*np.fft.fft(zlp.data, **kwfft)
    eels.data = np.fft.ifft(eels.data, **kwfft).real

    eels.get_dimensions_from_data()

    eels.data = np.roll(eels.data, -int(axis.offset/axis.scale), -1)
    eels.crop_signal1D(None, axis.size)
    return eels

def kk_fourier(im):
    """
    The energy axis is the last one.
    """
    re = np.zeros_like(im)
    esize = im.shape[-1]
    ids = (slice(None),) * (im.ndim-1) + (slice(None, esize),)
    dsize = 2 * esize
    q = - 2 * np.fft.fft(im, dsize, -1).imag / dsize
    q[ids] *= -1.
    q = np.fft.fft(q, axis=-1)
    return - q[ids].real

def chi_j_free_carrier(energy,
                       plasmon,
                       damping,
                       infy_value):
    """
    Calculate the susceptibility following the Drude model, containing
    a plasmon resonance.

    Parameters
    ----------
    energy : {float, np.ndarray}
     Energy-loss axis, contained in an array.
    plasmon : {float, np.ndarray}
     Band-gap energy, contained in an array.
    damping : float
    infy_value : float

    References
    ----------
    Drude model from Yu and Cardona 2011.
    """
    eax = energy[None,:]
    epl = plasmon[:, None]
    chi_j = infy_value * epl**2 * damping / (
            eax * (eax**2 + damping**2) )
    return chi_j

def chi_j_bound_carrier(energy,
                        binding,
                        plasmon,
                        damping,
                        infy_value):
    """
    Calculate the susceptibility following the Drude model, containing
    a plasmon resonance and a shift of the plasmon peak.

    Parameters
    ----------
    energy : {float, np.ndarray}
     Energy-loss axis, contained in an array.
    binding : {float, np.ndarray}
     Binding energy, contained in an array.
    plasmon : float
    damping : float
    infy_value : float

    References
    ----------
    Drude model from Egerton 2011.
    """
    eax = energy[None,:]
    ebd = binding[:, None]
    chi_j = infy_value * plasmon**2 * eax * damping / (
            (eax**2-ebd**2)**2 + (eax*damping)**2 )
    return chi_j

def chi_j_tauc_lorentz(energy,
                       bandgap,
                       resonance,
                       damping,
                       strength):
    """
    Calculate the susceptibility following the Tauc-Lorentz model, containing
    a plasma-like resonance that is 0 before the band gap energy.

    Parameters
    ----------
    energy : {float, np.ndarray}
     Energy-loss axis, contained in an array.
    bandgap : {float, np.ndarray}
     Band-gap energy, contained in an array.
    resonance : float
    strength : float
    damping : float

    References
    ----------
    Tauc-Lorenz model from Jellison APL 1996
    """
    eax = energy[None, :]
    gap = bandgap[:, None]
    resonance = resonance + gap
    step = eax - gap
    step[step<0] = 0
    step[step>0] = 1
    # Calculate separate plasmon
    chi_j = (1./eax)*(strength**2*resonance*damping*(eax-gap)**2) / (
             (eax**2-resonance**2)**2+eax**2*damping**2) * step
    return chi_j

def chi_j_tauc_bandgap(energy,
                       bandgap,
                       strength):
    """
    Calculate the susceptibility following the Tauc bandgap model, containing
    a direct allowed band-gap.

    Parameters
    ----------
    energy : {float, np.ndarray}
     Energy-loss axis, contained in an array.
    bandgap : {float, np.ndarray}
     Band-gap energy, contained in an array.
    resonance : float
    strength : float
    damping : float

    References
    ----------
    Tauc direct allowed bandgap, as presented in Yu and Cardona, 2011.
    """
    eax = energy[None, :]
    gap = bandgap[:, None]
    step = eax - gap
    step[step<0] = 0
    step[step>0] = 1
    # Calculate separate plasmon
    chi_j = strength**2 * np.sqrt(eax-gap) * step / eax**2
    return np.nan_to_num(chi_j)

def chi_j_reststrahlen(energy,
                       transverse,
                       gotdamp,
                       zero_value,
                       infy_value):
    '''
    gotdamp = damping / transverse, to preserve peak height.
    '''
    eax = energy[None,:]
    etr = transverse[:, None]
    damping = etr * gotdamp
    fdamp = (eax*damping/etr**2)
    chi_j = (zero_value - infy_value) * fdamp / (
            (1-(eax/etr)**2)**2 + fdamp**2)
    return chi_j

def dielectric_tauc_bandgap(energy,
                            bandgap,
                            thickness,
                            resonance=15.,
                            damping=0.05,
                            nf=1.,
                            nb=1.,
                            collection=10,
                            beam_energy=300):
    '''
    Dielectric function model for EELS, containing a free plasmon-like resonance
    and a Tauc direct allowed band gap resonance. This model is obtained by
    adding the susceptibilities of a Tauc-Lorenz model and a Tauc bandgap model.
    Such model corresponds to null absorption before the band gap onset and at
    infinite energy, which is nice.

    Parameters
    ----------
    energy : ndarray
     Defines the energy axis range of the calculated dielectric function.
    bandgap : ndarray
     Defines the band gap axis range of the calculated dielectric function.
    thickness : ndarray
     Defines the thickness axis range of the calculated dielectric function.
    resonance : float
     Free-electron-like resonance energy in eV, for example 15 eV.
    damping : float
     Life-time broadening of this resonance in eV, for instance 3 eV.
    nf : float
     Oscillator strength of the resonance. If set to 0, the resonance is not
     used.
    nb : float
     Oscillator strength of the direct band gap. If set to 0, a direct band gap
     is not used, but the band gap is still used to set the Tauc-Lorenz model.
    collection : float
     Set the collection angle. By default equal to 10 mrad.
    beam_energy : float
     Set the beam energy. By default equal to 300 keV.

    Returns
    -------
    eps : ModifiedCDF
     A model dielectric function, for testing purposes, with dimensions set by
     the input energy, bandgap and thickness.
    '''
    chi_j = np.zeros([len(bandgap), len(energy)])
    if nf != 0:
        chi_j += chi_j_tauc_lorentz(energy, bandgap, resonance, damping, nf)
    if nb != 0:
        chi_j += chi_j_tauc_bandgap(energy, bandgap, nb)
    eps =  1 + kk_fourier(chi_j) + 1j*chi_j
    eps = eps + 1j*1e-6

    t = np.repeat(thickness[None, :], len(bandgap), 0)
    eps = np.repeat(eps[:, None, :], len(thickness), 1)

    axdict = [{'name' : 'Eg, band gap',
              'size' : len(bandgap),
              'navigate' : True,
              'units' : 'eV',
              'scale' : bandgap[1]-bandgap[0],
              'offset' : bandgap[0]},
              {'name' : 't, thickness',
              'size' : len(thickness),
              'navigate' : True,
              'units' : 'nm',
              'scale' : thickness[1]-thickness[0],
              'offset' : thickness[0]},
              {'name' : 'Energy-loss',
              'size' : len(energy),
              'navigate' : False,
              'units' : 'eV',
              'scale' : energy[1]-energy[0],
              'offset' : energy[0]}]

    cdf = ModifiedCDF(eps, axes=axdict)
    beam = 'Acquisition_instrument.TEM.beam_energy'
    coll = 'Acquisition_instrument.TEM.Detector.EELS.collection_angle'
    cdf.metadata.set_item(beam, beam_energy)
    cdf.metadata.set_item(coll, collection)
    return cdf, t

def dielectric_reststrahlen(energy,
                            transverse,
                            thickness,
                            damping=0.01,
                            zero_value=5.,
                            infy_value=1.,
                            collection=10,
                            beam_energy=100.):
    '''
    Dielectric function model for EELS, containing a Restrahlen-type phonon
    resonance.

    Returns
    -------
    eps : ModifiedCDF
     A model dielectric function, for testing purposes, with dimensions set by
     the input energy, bandgap and thickness.
    '''
    chi_j = chi_j_reststrahlen(energy, transverse, damping, zero_value, infy_value)
    eps = infy_value + kk_fourier(chi_j) + 1j*chi_j
    eps = eps + 1j*1e-6

    t = np.repeat(thickness[None, :], len(transverse), 0)
    eps = np.repeat(eps[:, None, :], len(thickness), 1)

    axdict = [{'name' : r'$E_T$, phonon freq.',
              'size' : len(transverse),
              'navigate' : True,
              'units' : 'eV',
              'scale' : transverse[1]-transverse[0],
              'offset' : transverse[0]},
              {'name' : 't, thickness',
              'size' : len(thickness),
              'navigate' : True,
              'units' : 'nm',
              'scale' : thickness[1]-thickness[0],
              'offset' : thickness[0]},
              {'name' : 'Energy-loss',
              'size' : len(energy),
              'navigate' : False,
              'units' : 'eV',
              'scale' : energy[1]-energy[0],
              'offset' : energy[0]}]

    cdf = ModifiedCDF(eps, axes=axdict)
    beam = 'Acquisition_instrument.TEM.beam_energy'
    coll = 'Acquisition_instrument.TEM.Detector.EELS.collection_angle'
    cdf.metadata.set_item(beam, beam_energy)
    cdf.metadata.set_item(coll, collection)
    return cdf, t

def dielectric_drude_lorenz(energy,
                            plasmon,
                            thickness,
                            damping=0.6,
                            infy_value=1.,
                            collection=10.,
                            beam_energy=100.):
    '''
    Dielectric function model for EELS, containing a Free electron plasmon-type
    resonance.

    Returns
    -------
    eps : ModifiedCDF
     A model dielectric function, for testing purposes, with dimensions set by
     the input energy, bandgap and thickness.
    '''
    #chi_j = chi_j_free_carrier(energy, plasmon, damping, infy_value)
    #eps = infy_value + kk_fourier(chi_j) + 1j*chi_j
    eax = energy[None,:]
    epl = plasmon[:, None]
    eps = infy_value * (1 - epl**2 / (
            (eax**2 + 1j* damping*eax) ))
    eps = eps + 1j*1e-6

    t = np.repeat(thickness[None, :], len(plasmon), 0)
    eps = np.repeat(eps[:, None, :], len(plasmon), 1)

    axdict = [{'name' : r'$E_T$, phonon freq.',
              'size' : len(plasmon),
              'navigate' : True,
              'units' : 'eV',
              'scale' : plasmon[1]-plasmon[0],
              'offset' : plasmon[0]},
              {'name' : 't, thickness',
              'size' : len(thickness),
              'navigate' : True,
              'units' : 'nm',
              'scale' : thickness[1]-thickness[0],
              'offset' : thickness[0]},
              {'name' : 'Energy-loss',
              'size' : len(energy),
              'navigate' : False,
              'units' : 'eV',
              'scale' : energy[1]-energy[0],
              'offset' : energy[0]}]

    cdf = ModifiedCDF(eps, axes=axdict)
    beam = 'Acquisition_instrument.TEM.beam_energy'
    coll = 'Acquisition_instrument.TEM.Detector.EELS.collection_angle'
    cdf.metadata.set_item(beam, beam_energy)
    cdf.metadata.set_item(coll, collection)
    return cdf, t
