import numpy as np
import math

from hyperspy.component import Component

sqrt2pi = math.sqrt(2 * math.pi)

def _voigt(x, FWHM=1, gamma=1, center=0, scale=1):
    """Voigt lineshape.

    The Voigt peak is the convolution of a Lorentz peak with a Gaussian peak.

    The formula used to calculate this is::

        z(x) = (x + 1j gamma) / (sqrt(2) sigma)
        w(z) = exp(-z**2) erfc(-1j z) / (sqrt(2 pi) sigma)

        V(x) = scale Re(w(z(x-center)))

    Parameters
    ----------
    gamma : real
       The half-width half-maximum of the Lorentzian
    FWHM : real
       The FWHM of the Gaussian
    center : real
       Location of the center of the peak
    scale : real
       Value at the highest point of the peak

    Notes
    -----
    Ref: W.I.F. David, J. Appl. Cryst. (1986). 19, 63-64

    adjusted to use stddev and HWHM rather than FWHM parameters

    """
    # wofz function = w(z) = Fad[d][e][y]eva function = exp(-z**2)erfc(-iz)
    from scipy.special import wofz
    sigma = FWHM / 2.3548200450309493
    z = (np.asarray(x) - center + 1j * gamma) / (sigma * math.sqrt(2))
    V = wofz(z) / (math.sqrt(2 * np.pi) * sigma)
    return scale * V.real


class ZeroLossPeak(Component):

    """Modified Voigt profile component with support for measuring the Shirley
    background; additionally non_isochromaticity, transmission_function
    corrections and spin orbit splitting make this component specially suited
    for EELS and PES data analysis.

    f(x) = G(x)*L(x) where G(x) is the Gaussian function and L(x) is the
    Lorentzian function

    Attributes
    ----------

    area : Parameter
    centre: Parameter
    FWHM : Parameter
    gamma : Parameter
    resolution : Parameter
    shirley_background : Parameter
    non_isochromaticity : Parameter
    transmission_function : Parameter
    spin_orbit_splitting : Bool
    spin_orbit_branching_ratio : float
    spin_orbit_splitting_energy : float

    """

    def __init__(self):
        Component.__init__(self, (
            'area',
            'centre',
            'FWHM',
            'gamma',
            'resolution',
            'background',
            'non_isochromaticity',
            'transmission_function',
            'offset'))
        self._position = self.centre
        self.FWHM.value = 1
        self.gamma.value = 0
        self.area.value = 1
        self.resolution.value = 0
        self.resolution.free = False
        self.background.value = 0.
        self.background.free = False
        self.non_isochromaticity.value = 0
        self.non_isochromaticity.free = False
        self.transmission_function.value = 1
        self.transmission_function.free = False

        # Options
        self.background.active = False
        self.spin_orbit_splitting = False
        self.spin_orbit_branching_ratio = 0.5
        self.spin_orbit_splitting_energy = 0.61

        self.isbackground = False
        self.convolved = True

    def function(self, x):
        area   = self.area.value * self.transmission_function.value
        centre = self.centre.value
        ab     = self.non_isochromaticity.value
        if self.resolution.value == 0:
            FWHM = self.FWHM.value
        else:
            FWHM = math.sqrt(self.FWHM.value ** 2 + self.resolution.value ** 2)
        gamma = self.gamma.value
        kvoigt = {'x'      : x,
                  'FWHM'   : FWHM,
                  'gamma'  : gamma,
                  'center' : centre-ab,
                  'scale'  : area}
        f = _voigt(**kvoigt)

        if self.spin_orbit_splitting is True:
            ratio = self.spin_orbit_branching_ratio
            shift = self.spin_orbit_splitting_energy
            kvoigt['center'] = centre-ab-shift
            kvoigt['scale']  = area * ratio
            f = _voigt(**kvoigt)
            f += f2

        if self.background.active:
            k = self.background.value
            cf = np.cumsum(f)
            cf = cf[-1] - cf
            self.cf = cf
            return cf * k + f
        else:
            k = self.background.value
            return k + f

class TaucBandGap(Component):

    """A Tauc direct Band Gap

    """

    def __init__(self, gap_energy=1., strength=1.):
        Component.__init__(self, ('gap_energy', 'strength'))
        self.gap_energy.value = gap_energy
        self.strength.value = strength
        self.isbackground = False
        self.convolved = False

    def function(self, x):
        x = np.asanyarray(x)
        gap_energy = self.gap_energy.value
        strength = self.strength.value
        # Heaviside step
        out = x - gap_energy
        step = out.copy()
        step[step<0] = 0
        step[step>0] = 1
        # Tauc direct bandgap
        out = np.nan_to_num(
            strength**2 * np.sqrt(out) * step / x**2)
        return out

class TaucLorentzResonance(Component):

    """A Tauc-Lorentz shifted resonance

    """

    def __init__(self,
                 gap_energy=1.,
                 resonance_energy=5.,
                 damping=1.,
                 strength=1.):
        Component.__init__(self,
                           ('gap_energy',
                            'resonance_energy',
                            'damping',
                            'strength'))
        self.gap_energy.value = gap_energy
        self.resonance_energy.value = resonance_energy
        self.damping.value = damping
        self.strength.value = strength
        self.isbackground = False
        self.convolved = False

    def function(self, x):
        x = np.asanyarray(x)
        gap_energy = self.gap_energy.value
        resonance_energy = self.resonance_energy.value + gap_energy
        damping = self.damping.value
        strength = self.strength.value
        # Heaviside step
        out = x - gap_energy
        step = out.copy()
        step[step<0] = 0
        step[step>0] = 1
        # Jellison APL 1996
        out = (1./x)*(strength**2*resonance_energy*damping*(out)**2) / (
             (x**2-resonance_energy**2)**2+x**2*damping**2) * step
        return out