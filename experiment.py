'''
This file contains scripts to automate analysis of experimental EELS. Power-law
and mirror zero-loss tail extrapolation tools are included. The deconvolved
single scattering distribution can be used together with the relativistic
Kramers-Kronig wrapper to obtain the underlying complex dielectric function.
'''
import hyperspy.api as hs
import numpy as np

# own dependencies
from dielectric import ModifiedCDF
from eels import  ModifiedEELS

from glob import glob

def regex_to_names(str_regex):
    '''Use a regular expression (regex) to generate a list of global addresses
    from files in disk. This list of strings can be used to load data and so on.

    Parameters
    ----------
    str_regex : string
     Containing the regex that we want to use. This is transformed into global
     addresses by `glob.glob`, and sorted using `sorted` builtin.

    Returns
    -------
    names : list of strings
     A list of global addresses of files in disc.
    '''
    names = sorted(glob(str_regex))
    return names

def load_eels(name,
              beam_energy=None,
              collection=None,
              convergence=None,
              align=True):
    '''Load EELS data, set the appropriate parameters and register it using the
    hyperspy methods; `load`, `set_microscope_parameters` and
    `align_zero_loss_peak`, respectively.

    Parameters
    ----------
    name : str
     Relative or global address of the file in disk.
    beam_energy : float
    collection : float
    convergence : float
    align : bool
     Self-explanatory

    Returns
    -------
    s : ModifiedEELS
     The registered EELS dataset.
    '''
    s = hs.load(name, lazy=False)
    s  = ModifiedEELS(s)

    if beam_energy is not None:
        str = 'Acquisition_instrument.TEM.beam_energy'
        s.metadata.set_item(str, beam_energy)
    if collection is not None:
        str = 'Acquisition_instrument.TEM.Detector.EELS.collection_angle'
        s.metadata.set_item(str, collection)
    if convergence is not None:
        str = 'Acquisition_instrument.TEM.convergence_angle'
        s.metadata.set_item(str, convergence)

    if align:
        s.align_zero_loss_peak(print_stats=False, show_progressbar=False)
    return s

def run_zlp_model(Iini,
                  zlp_tail_range=(0.5, 1.),
                  ssd_crop_range=(1., 40.),
                  spc_tail_extra=(100, 2048),
                  spectral_rebin=None):
    """ Model the zero-loss peak tail using a power-law extrapolation. Abridged.

    The power-law extrapolation is the workhorse of EELS analysis. The zero-loss
    tail is extrapolated from a fit to an energy range. The resulting model can
    be used to remove elastic and plural scattering from the EELS dataset. This
    is done using `fourier_log_deconvolution`. Previously, the high-energy tails
    of all datasets are smoothly extrapolated to zero, using again a power-law.

    Parameters
    ----------
    Iini : ModifiedEELS
     Input EELS dataset. Should've been previously registered, see `load_eels`.
    zlp_tail_range : tuple
     Two float values used for the zero-loss tail power-law extrapolation.
    ssd_crop_range : tuple
     Two float values setting the range of the inelastic scattering information.
    spc_tail_extra : tuple
     Two integer values used for the inelastic scattering tail extrapolation.
    spectral_rebin : int
     Set this to rebin the spectral axis by the specified scale.

    Returns
    -------
    Issd, Izlp : ModifiedEELS
     Deconvolved single scattering distribution (SSD) and zero-loss peak (ZLP).
    """

    E1, E2 = zlp_tail_range     # Power law ZLP tail model
    Xini, Xfin = ssd_crop_range # Prepare SSD for KKA
    Wspc, Nspc = spc_tail_extra # High-energy tail extrapolations

    # Optionally... (not used here)
    Wkka = Wspc #... use a different power_law_extrapolation for kka
    Nkka = Nspc
    Ecut = None #... remove the EELS intensity before Ecut for kka

    # Power law ZLP tail model + fl-deconvolution
    Izlp = Iini.model_zero_loss_peak_tail((E1, E2), fast=False, show_progressbar=False)
    Izlp = Izlp.power_law_extrapolation_until(Wspc, Nspc, fix_neg_r=True, add_noise=True)
    Iexp = Iini.power_law_extrapolation_until(Wspc, Nspc, fix_neg_r=True, add_noise=True)
    Ifld = Iexp.fourier_log_deconvolution(Izlp)

    # Prepare SSD for KKA
    Issd = Ifld.remove_negative_intensity()
    Issd.crop_signal1D(Xini, Xfin)
    Issd = Issd.power_law_extrapolation_until(Wkka, Nkka, True, fix_neg_r=True, add_noise=True)
    if Ecut is not None:
        Issd.remove_intensity(None, Ecut, True)
    Issd.data += 1e-6

    # Spectral rebin
    if spectral_rebin is not None:
        axis = Issd.axes_manager.signal_axes[0]
        idt = (1,) * axis.index_in_array + (spectral_rebin,)
        Issd = Issd.rebin(scale=idt)
        Izlp = Izlp.rebin(scale=idt)

    return Issd, Izlp

def run_zlp_mirror(Iini,
                   zlp_tail_cut=1.,
                   ssd_crop_range=(1., 40.),
                   spc_tail_extra=(100, 2048),
                   spectral_rebin=None):
    """ Model the zero-loss peak tail using a mirror extrapolation. Abridged.

    The power-law extrapolation is the stuff of dreams in EELS analysis.  In a
    monochromated instrument the zero-loss peak is allegedly symmetric. Hence,
    the positive and negative energy-loss tail can be interchanged. In this
    script, this is done only starting from a given energy value that should be
    set according to the intersection of both tails. The resulting model can be
    used to remove elastic and plural scattering from the EELS dataset. This is
    done using `fourier_log_deconvolution`. Previously, the high-energy tails of
    all datasets are smoothly extrapolated to zero, using again a power-law.

    Parameters
    ----------
    Iini : ModifiedEELS
     Input EELS dataset. Should've been previously registered, see `load_eels`.
    zlp_tail_cut : float
     The positive energy-loss data is preferred until this value.
    ssd_crop_range : tuple
     Two float values setting the range of the inelastic scattering information.
    spc_tail_extra : tuple
     Two integer values used for the inelastic scattering tail extrapolation.
    spectral_rebin : int
     Set this to rebin the spectral axis by the specified scale.

    Returns
    -------
    Issd, Izlp : ModifiedEELS
     Deconvolved single scattering distribution (SSD) and zero-loss peak (ZLP).
    """

    Xini, Xfin = ssd_crop_range # Prepare SSD for KKA
    Wspc, Nspc = spc_tail_extra # High-energy tail extrapolations

    # Optionally... (not used here)
    Wkka = Wspc #... use a different power_law_extrapolation for kka
    Nkka = Nspc
    Ecut = None #... remove the EELS intensity before Ecut for kka

    # Power law ZLP tail model + fl-deconvolution
    Izlp = Iini.model_zero_loss_peak_mirror()
    Izlp = Izlp.power_law_extrapolation_until(Wspc, Nspc, fix_neg_r=True, add_noise=True)
    Iexp = Iini.power_law_extrapolation_until(Wspc, Nspc, fix_neg_r=True, add_noise=True)

    # The positive energy data is preferred until zlp_tail_cut!
    axis = Izlp.axes_manager.signal_axes[0]
    idx = axis.value2index(zlp_tail_cut)
    ids = (slice(None),) * axis.index_in_array + (slice(None, idx), Ellipsis)
    Izlp.data[ids] = Iexp.data[ids]

    Ifld = Iexp.fourier_log_deconvolution(Izlp)

    # Prepare SSD for KKA
    Issd = Ifld.remove_negative_intensity()
    Issd.crop_signal1D(Xini, Xfin)
    Issd = Issd.power_law_extrapolation_until(Wkka, Nkka, True, fix_neg_r=True, add_noise=True)
    if Ecut is not None:
        Issd.remove_intensity(None, Ecut, True)
    Issd.data += 1e-6

    # Spectral rebin
    if spectral_rebin is not None:
        idt = (1,) * axis.index_in_array + (spectral_rebin,)
        Issd = Issd.rebin(scale=idt)
        Izlp = Izlp.rebin(scale=idt)

    return Issd, Izlp

def run_kka(Issd, Izlp, get_chi2_score=True, *args, **kwargs):
    """Run relativistic Kramers-Kronig analysis from an input single scattering
    distribution (SSD) and zero-loss peak (ZLP).

    Parameters
    ----------
    Issd, Izlp : ModifiedEELS
     The single scattering distribution (SSD) dataset and a zero-loss peak (ZLP)
     dataset, modelled or intensity only.
    *args, **kwargs passed on to `Issd.relativistic_kramers_kronig`, see docs
    there for more info.

    Returns
    -------
    eps : ModifiedCDF
     Complex dielectric function (CDF) obtained from the analysis.
    out : Dictionary
     Containing potentially two entries: out['relativist correction'] and
     out['thickness'].
    """
    eps, out = Issd.relativistic_kramers_kronig(Izlp, *args, **kwargs)
    t = out['thickness']
    print('Thickness / nm :', t.data.mean())
    if get_chi2_score:
        Itot, Ivol, Ielf = eps.get_relativistic_spectrum(Izlp, t)
        chi2 = Issd._chi2_score(Itot)
        print('Xi-squared evl :',  chi2)
    return eps, out

## Apply again the ZLP fit and extract the SSD


def extract_ssd(spc, zlp, kwpad=None):
    """
    Perform Fourier-log deconvolution using a zlp model.

    Parameters
    ----------
    spc : ModifiedEELS
     The input EEL spectra
    zlp : hyperspy model
     Input for spc.model_zero_loss_peak
    kwpad : {None, dictionary}
     Optionally containing the input parameters for the pad. This is performed
     using the power_law_extrapolation_until method.

    Returns
    -------
    ssd, z : ModifiedEELS
     Single scattering distribution and corresponding ZLP model.
    """

    zlp_threshold = float(zlp.axis.axis[zlp.channel_switches][-1])
    axis = spc.axes_manager[-1]
    ssd_range = (float(axis.scale), float(axis.high_value+axis.scale))

    # Update ZLP model with appropriate size (expand and crop technique)
    z = spc.model_zero_loss_peak(threshold = zlp_threshold,
                                 model     = zlp,
                                 show_progressbar = False)
    ssd = spc.deepcopy()
    if kwpad is not None:
        # Pad the EELS spectra
        ssd = ssd.power_law_extrapolation_until(**kwpad)

    # extract SSD using the Fourier-log deconvolution
    ssd = ssd.fourier_log_deconvolution(z)
    ssd.crop_signal1D(*ssd_range)
    ssd.remove_negative_intensity()
    return ssd, z

def process_ssd(ssd, left, right, hanning_width=None):
    """
    Calculates the intensity offset from a region of the provided SSD and
    removes it.

    Parameters
    ----------
    left, right : float
     Define the range in the signal axis units.
    hanning_width : {None, float}
     Optionally apply a hanning taper to the left side with the provided width.

    Returns
    -------
    ssd, background : hyperspy Signals
     The processed ssd and subtracted intensity background.
    """

    axis = ssd.axes_manager.signal_axes[0]
    middle = left + (right-left) / 2.

    background = ssd.isig[left:right].mean(-1)
    int_middle = ssd.axes_manager[-1].value2index(middle)

    ssd_p = ssd.__class__(ssd - background)
    if hanning_width is not None:
        ssd_p.hanning_taper('left', channels=hanning_width, offset=int_middle)

    return ssd_p, background

def calculate_eps(ssd, z, n, kwpad=None, background=None):
    """
    Calculates the dielectric function for the provided ZLP and refractive idx.

    Parameters
    ----------
    ssd, z : ModifiedEELS
     Single scattering distribution and zero-loss peak datasets.
    n : float
     Refractive index.

    Returns
    -------
    eps, thickness: hyperspy Signals
     The dielectric function and corresponding thickness.
    """

    axis = ssd.axes_manager.signal_axes[0]
    ssd_range = (float(axis.scale), float(axis.high_value+axis.scale))

    if background is not None:
        Izlp = (z - background).integrate1D(-1)
    else:
        Izlp = z.integrate1D(-1)

    if kwpad is not None:
        # Pad the EELS spectra
        ssd_p = ssd.power_law_extrapolation_until(**kwpad)

    elf, thickness = ssd_p.normalize_bulk_inelastic_scattering(Izlp, n=n)
    eps = elf.kramers_kronig_transform(invert=True)
    eps.crop(-1, *ssd_range)

    return eps, thickness
