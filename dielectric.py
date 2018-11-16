import numpy as np
from numbers import Number
from scipy import constants
from scipy.integrate import simps, cumtrapz
from integration import interp_quad, interp_cuba, log_sum_exp

# hyperspy dependency
import hyperspy.api as hs

import logging
from hyperspy.signal import _logger
from hyperspy.misc import physical_constants as cttes

# own dependency
from signals import SignalMixin

class ModifiedCDF(hs.signals.DielectricFunction, SignalMixin):

    def __init__(self, *args, **kwargs):
        """
        Modified hypespy CDF signal. Input can be array or DielectricFunction.
        In the last case, additional *args and **kwargs are discarded.
        """
        if len(args) > 0 and isinstance(args[0], hs.signals.DielectricFunction):
            # Pretend it is a hs signal, copy axes and metadata
            sdict = args[0]._to_dictionary()
            hs.signals.DielectricFunction.__init__(self, **sdict)
        else:
            hs.signals.DielectricFunction.__init__(self, *args, **kwargs)

    def get_cerenkov_angle(self):
        '''
        Calculates the characteristic Cerenkov scattering angle, following eq.
        3.70 in [Egerton2011]_. This quantity can be used to estimate the
        position of the Lorentzian-like maxima of the bulk relativistic losses
        in the scattering angle vs. evergy graph of the DDCS.

        Returns
        -------
        ThetaC : ComplexSignal1D
         The Cerenkov scattering angle as a function of energy, which is a
         complex quantity. The amplitude can be examined by calling `amplitude`.
         The phase indicates branch cuts, and is easily examined using `phase`.

        See also
        --------
        get_relativistic_ddcs

        '''
        # Define some constants
        mel  = cttes.m0  # Electron mass in kg
        e    = cttes.e   # Electron charge in Coulomb
        c    = 2.99792458e8  # speed of light m/s

        # read beam energy from meatadata
        metadata = self.metadata
        beam = 'Acquisition_instrument.TEM.beam_energy'
        if metadata.has_item(beam):
            beam_energy = metadata.get_item(beam)
        else:
            raise ValueError('Beam energy not defined. It can be ' + \
                             'defined using set_microscope_parameters')

        # Read the energy axis
        energies = self.axes_manager[-1].axis.copy()
        epsilon_real = self.data.real

        va = 1. - (511. /(511. + beam_energy))**2.
        v  = c * np.sqrt(va)
        beta2 = (v / c)**2
        gamma = 1. / np.sqrt(1. - beta2)
        momentum = mel * v * gamma

        # this might just broadcast on his own, which is great
        adata = energies * e / (momentum * c) * \
                                       np.lib.scimath.sqrt(epsilon_real - beta2)
        ThetaC = self._deepcopy_with_new_data(data=adata)
        ThetaC.set_signal_type('ComplexSignal1D')
        ThetaC.metadata.General.title = (self.metadata.General.title +
                                         r', $\theta_C$')
        return ThetaC

    def get_characteristic_angle(self):
        '''
        Calculates the characteristic scattering angle, following eq. 3.28 in
        [Egerton2011]_. This quantity indicates the position of the Lorentzian
        maxima of bulk semi-relativistic losses in the scattering angle vs.
        energy graph of the DDCS.

        Returns
        -------
        ThetaE : Signal1D
         The characteristic scattering angle as a function of energy.

        See also
        --------
        get_relativistic_ddcss
        '''
        # Define some constants
        mel  = cttes.m0  # Electron mass in kg
        e    = cttes.e   # Electron charge in Coulomb
        c    = 2.99792458e8  # speed of light m/s

        # read beam energy from meatadata
        metadata = self.metadata
        beam = 'Acquisition_instrument.TEM.beam_energy'
        if metadata.has_item(beam):
            beam_energy = metadata.get_item(beam)
        else:
            raise ValueError('Beam energy not defined. It can be ' + \
                             'defined using set_microscope_parameters')

        # Read the energy axis
        energies = self.axes_manager[-1].axis.copy()

        va = 1. - (511. /(511. + beam_energy))**2.
        v  = c * np.sqrt(va)
        beta2 = (v / c)**2
        gamma = 1. / np.sqrt(1. - beta2)
        momentum = mel * v * gamma

        # this might just broadcast on his own, which is great
        adata = energies * e / momentum / v
        ThetaE = self._get_signal_signal(data=adata).real
        ThetaE.metadata.General.title = (self.metadata.General.title +
                                         r', $\theta_E$')
        return ThetaE

    def get_number_of_effective_electrons(self, nat=None, plasmon_energy=None,
                                          cumulative=True):
        r"""Compute the number of effective electrons using the Bethe f-sum
        rule.

        The Bethe f-sum rule gives rise to two definitions of the effective
        number (see [Egerton2011]_), neff1 and neff2:

            .. math::

                n_{\mathrm{eff_{1}}} = n_{\mathrm{eff}}\left(-\Im\left(\epsilon^{-1}\right)\right)

        and:

            .. math::

                n_{\mathrm{eff_{2}}} = n_{\mathrm{eff}}\left(\epsilon_{2}\right)

        This method computes and return both.

        Parameters
        ----------
        nat: {None, float}
            Provide the number of atoms (or molecules) per unit volume of the
            sample. Can be None, then plasmon_energy must be provided.
        plasmon_energy : {None, float}
            Alternatively provide the free electron resonance energy for this
            dielectric function. Both nat and plasmon_energy should not be given
        cumulative : bool
            If False calculate the number of effective electrons up to the
            higher energy-loss of the spectrum. If True, calculate the
            number of effective electrons as a function of the energy-loss up
            to the higher energy-loss of the spectrum. *True is only supported
            by SciPy newer than 0.13.2*.

        Returns
        -------
        neff1, neff2: Signal1D
            Signal1D instances containing neff1 and neff2. The signal and
            navigation dimensions are the same as the current signal if
            `cumulative` is True, otherwise the signal dimension is 0
            and the navigation dimension is the same as the current
            signal.

        Notes
        -----
        .. [Egerton2011] Ray Egerton, "Electron Energy-Loss
        Spectroscopy in the Electron Microscope", Springer-Verlag, 2011.

        """

        if plasmon_energy is None and nat is not None:
            m0 = constants.value("electron mass")
            epsilon0 = constants.epsilon_0    # Vacuum permittivity [F/m]
            hbar = constants.hbar             # Reduced Plank constant [J·s]
            k = 2 * epsilon0 * m0 / (np.pi * nat * hbar ** 2)
        elif plasmon_energy is not None and nat is None:
            k = 8*(np.pi*plasmon_energy**2)**-1
        else:
            raise AttribureError("Either nat or plasmon_energy should be given,"
                                 " just one of them, not both parameters.")

        axis = self.axes_manager.signal_axes[0]
        if cumulative is False:
            dneff1 = k * simps((-1. / self.data).imag * axis.axis,
                               x=axis.axis,
                               axis=axis.index_in_array)
            dneff2 = k * simps(self.data.imag * axis.axis,
                               x=axis.axis,
                               axis=axis.index_in_array)
            neff1 = self._get_navigation_signal(data=dneff1)
            neff2 = self._get_navigation_signal(data=dneff2)
        else:
            neff1 = self._deepcopy_with_new_data(
                k * cumtrapz((-1. / self.data).imag * axis.axis,
                             x=axis.axis,
                             axis=axis.index_in_array,
                             initial=0))
            neff2 = self._deepcopy_with_new_data(
                k * cumtrapz(self.data.imag * axis.axis,
                             x=axis.axis,
                             axis=axis.index_in_array,
                             initial=0))

        # Prepare return
        neff1.metadata.General.title = (r'$n_{eff}$ from energy-loss')
        neff2.metadata.General.title = (r'$n_{eff}$ from absorption')

        return neff1, neff2

    def get_complex_refractive_index(self):
        nn = self.deepcopy()
        nn.data = np.lib.scimath.sqrt(nn.data)
        nn.metadata.General.title = 'Complex refractive index'
        return nn

    def get_reflection_coeff(self):
        nn = self.get_complex_refractive_index()
        rcoeff = ((nn.real-1)**2+nn.imag**2) / ((nn.real+1)**2+nn.imag**2)
        rcoeff.metadata.General.title = 'Reflection coefficient'
        return rcoeff

    def get_absorption_coeff(self):
        e    = cttes.e       # Electron charge in Coulomb
        c    = 2.99792458e8  # speed of light m/s
        hbar = 1.0545718e-34 # Planck's constant

        nn = self.get_complex_refractive_index()
        eax = self.axes_manager.signal_axes[0].axis
        acoeff = 2 * eax * e * nn.imag / (hbar * c)
        acoeff.metadata.General.title = r'Absorption coefficient / $m^{-1}$'
        return acoeff

    def get_relativistic_ddcs(self, t, eta=1, NA=256, theta_min=1e-3,
                              show_progressbar=False, samlog=True):
        '''
        Calculate the double differential cross section corresponding to this
        dielectric function in a thin slab experiment using Kroeger formalism
        for normal incidence. The thin foil is covered by a loss-less medium,
        such as vacuum or optionally defined by either a dielectric constant or
        a complex dielectric function.

        The full relativistic DDCS (bulk + surface), and the relativistic and
        non-relativistic bulk terms are returned.

        Parameters
        ----------
        t : {number, ndarray, signal}
         The slab thickness in nm. Must be defined in order to obtain the proper
         relativistic DDCS surface terms.
        eta : {number, ModifiedCDF}
         Used to define the surrounding dielectric medium. No bulk loss is
         included for this medium. By default it is set equal to 1, for an empty
         surrounding medium. Can be set to a number o numpy array with the same
         number of elements as the input dielectric function energy axis.
        NA : number
         Define the size of the logarithmic scattering angule mesh, which
         determines the size of the output. By default set equal to 256.
        theta_min : float
         Define the minimum scattering angle in mrad, which together with the
         collection angle, determines the range of the logarithmic scattering
         angle mesh. By default set to 1 µrad.
        show_progressbar : {None, bool}
        samlog : bool
         If True, use a logarithmic scattering angle mesh (default). It can be
         set to False, and use a linear mesh for testing purposes.

        Returns
        -------
        Prel_tot, Prel_vol, Pelf : (list of 3) Signal2D
         The full relativistic DDCS accompanied of the relativistic and semi-
         relativistic bulk terms.
        '''
        # Avoid singularity at 0
        s = self.deepcopy()
        if self.axes_manager.signal_axes[0].axis[0] == 0:
            s = s.isig[1:]
        axis = s.axes_manager.signal_axes[0]
        energies = axis.axis.copy()
        NE = len(energies)

        thk = t*1. # failsafe copy mechanism ftw
        thk = s._check_adapt_map_input(thk)
        if isinstance(thk, ValueError):
            thk.args = (thk.args[0]+'t',)
            raise thk
        thk.data = thk.data * 1e-9 # Thickness input to m

        # read beam energy from meatadata
        metadata = s.metadata
        beam = 'Acquisition_instrument.TEM.beam_energy'
        if metadata.has_item(beam):
            beam_energy = metadata.get_item(beam)
        else:
            raise ValueError('Beam energy not defined. It can be ' + \
                             'defined using set_microscope_parameters')

        # same for collection angle
        coll = 'Acquisition_instrument.TEM.Detector.EELS.collection_angle'
        if metadata.has_item(coll):
            collection_angles = (theta_min, metadata.get_item(coll)) # it is in mrad
        else:
            raise ValueError('Collection angle not defined. It can be ' + \
                             'defined using set_microscope_parameters')

        a1 = collection_angles[0]*1e-3
        a2 = collection_angles[1]*1e-3

        if samlog:
            # generate a log-mesh of collection angles in rad
            angles = np.geomspace(a1, a2, NA)
            angles_axis = np.log10(angles) # in order to make a linear axis
            unitstr = 'log10(rad)'
            axisstr = 'log10(Scattering angle)'
        else:
            # generate a linear mesh of collection angles in rad
            angles = np.linspace(a1, a2, NA)
            angles_axis = angles # dummy
            unitstr = 'rad'
            axisstr = 'Scattering angle'

        # Define some constants
        mel  = cttes.m0  # Electron mass in kg
        bohr = cttes.a0  # Bohr radius in meters
        e    = cttes.e   # Electron charge in Coulomb
        c    = 2.99792458e8  # speed of light m/s
        h    = 6.626068e-34  # Planck's constant
        hbar = h / 2. / np.pi

        #Calculate fixed terms of equation
        va = 1. - (511. /(511. + beam_energy))**2.
        v  = c * np.sqrt(va)
        beta = v / c
        beta2 = beta**2.
        gamma = 1. / np.sqrt(1. - beta2)
        momentum = mel * v * gamma

        # Define some of the meshes for the calculation
        E, Theta   = np.meshgrid( energies, angles )
        Theta2 = Theta**2 + 1.0e-15
        ThetaE = E * e / momentum / v
        ThetaE2 = ThetaE**2

        # Covering layer: eta dependent terms
        if isinstance(eta, Number):
            eta = np.conj(eta)
            #lmb2_eta = Theta2 - eta * ThetaE2 * beta2
            #lmb_eta  = np.lib.scimath.sqrt(lmb2_eta)
            #phi2_eta = lmb2_eta + ThetaE2
        elif isinstance(eta, ModifiedCDF):
            # check eta navigation
            dummy = self._check_adapt_map_input(eta._get_navigation_signal())
            if isinstance(dummy, ValueError):
                dummy.args = (dummy.args[0]+'eta',)
                raise dummy
            # check eta signal
            if eta.axes_manager.signal_shape != self.axes_manager.signal_shape:
                raise ValueError('eta parameter error! signal dimension should '
                                 'coincide with the one from self.')
            # remove zero
            if eta.axes_manager.signal_axes[0].axis[0] == 0:
                eta = eta.isig[1:]
            eta.data = np.conj(eta)
        else:
            raise ValueError('eta parameter error! The provided covering layer '
                             'has to be a number or ModifiedCDF.')

        Pcoef = e / (bohr*np.pi**2.*mel*v**2.)
        Psurf = energies * e / 2.0 / hbar / v
        Psurf = Psurf[None, :].repeat(NA, 0)

        #... for the input/output (final calculation done inplace)
        iomesh = s.data.conj().astype('complex128')
        iomesh = np.repeat(iomesh[..., None,:], NA, -2)

        axdict = s.axes_manager.as_dictionary()
        axlist = [axdict[key] for key in axdict.keys()]
        axlist.insert(-1, {'name' : axisstr,
                           'size' : NA,
                           'navigate' : False,
                           'units' : unitstr,
                           'scale' : angles_axis[1]-angles_axis[0],
                           'offset' : angles_axis[0]} )

        iosig = hs.signals.ComplexSignal2D(
                    iomesh, axes=axlist, metadata=s.metadata.as_dictionary())

        def _semi_relativistic_loss(eps):
            """
             Builds the semi-relativistic bulk loss (Ritchie)
            """
            P = Pcoef * np.imag(1./eps) / np.real(Theta**2.+ThetaE**2.)
            return P

        def _full_relativistic_loss(eps, eta, t):
            """
             Builds the full-relativistic loss (Kroeger)
            """
            # Surrounding medium, eta terms for surface loss
            lmb2_eta = Theta2 - eta * ThetaE2 * beta2
            lmb_eta  = np.lib.scimath.sqrt(lmb2_eta)
            phi2_eta = lmb2_eta + ThetaE2

            # Thin layer, epsilon terms for surface loss
            lmb2_eps = Theta2 - eps * ThetaE2 * beta2
            lmb_eps  = np.lib.scimath.sqrt(lmb2_eps) # should be > 0.
            phi2_eps = lmb2_eps + ThetaE2

            # Combined term for relativistic surface loss
            phi2_eps_eta = Theta2 + ThetaE2 * (1. - (eps + eta) * beta2)

            # Thickness dependent terms for surface loss
            de = t * Psurf
            sin_de = np.sin(de)
            cos_de = np.cos(de)
            txy = np.tanh(lmb_eps * de / ThetaE)
            lplus = lmb_eta * eps + lmb_eps * eta * txy
            lminus = lmb_eta * eps + lmb_eps * eta / txy

            # "Relativistic surface plasmon"
            A1 = phi2_eps_eta**2. / eps / eta
            A2 = sin_de**2. / lplus + cos_de**2. / lminus
            A = A1 * A2
            # Guided light mode 1
            B1 = beta2 * lmb_eta * ThetaE * phi2_eps_eta / eta
            B2 = (1. / lplus - 1. / lminus) * 2. * sin_de * cos_de
            B = B1*B2
            # Guided light mode 2
            C1 = - beta2**2. * lmb_eta * lmb_eps * ThetaE2
            C2 = cos_de**2. * txy / lplus
            C3 = sin_de**2. / txy / lminus
            C = C1 * (C2 + C3)

            # Build relativistic surface loss
            Ps1 = 2 * Theta2 * (eps - eta)**2. / phi2_eta**2. / phi2_eps**2.
            Ps2 = hbar/momentum
            Ps3 = A + B + C
            Ps = Ps1 * Ps2 * Ps3

            # Build relativistic bulk loss
            Pv = t * (1. - (eps*beta2)) / eps / phi2_eps

            # Calculate P and Pvol (volume only)
            P = Pcoef * np.imag(Pv - Ps)
            Pvol = Pcoef * np.imag(Pv)
            return (P + 1j*Pvol)

        _logger.setLevel(logging.CRITICAL)

        # Map semi-relativistic loss (not inplace, fast)
        Pelf = iosig.map(_semi_relativistic_loss, inplace=False,
                                 show_progressbar=show_progressbar)
        Pelf = Pelf * thk if len(self) != 1 else Pelf.real * thk.data[0]
        Pelf.metadata.General.title = 'Semi-relativistic bulk loss'

        # Map full-relativistic loss (inplace, slow)
        iosig.map(_full_relativistic_loss, eta=eta, t=thk, inplace=True,
                                 show_progressbar=show_progressbar)

        Prel_tot = iosig.real
        Prel_tot.metadata.General.title = 'Full-relativistic loss (Bulk+Surf)'

        Prel_vol = iosig.imag
        Prel_vol.metadata.General.title = 'Full-relativistic loss (Bulk term)'

        ret = [Prel_tot, Prel_vol, Pelf]

        for sig in ret:
            del sig.metadata.Signal.signal_type

        return ret

    def get_relativistic_spectrum(self,
                                  zlp,
                                  t,
                                  output='full',
                                  ddcs=None,
                                  method='simpson',
                                  Ncpu = 4,
                                  show_progressbar=False,
                                  *args, **kwargs):
        '''
        Calculate the relativistic EELS from this dielectric function. The
        spectra are obtained by integrating the relativistic double differetial
        cross section. This is obtained calling `self.get_relativistic_ddcs`.
        The integration is then performed using either Simpson or quad method.

        Parameters
        ----------
        zlp : {number, ndarray, signal}
         Zero-loss peak intensity or signal. Navigation dimensions have to be 1
         or the same as self. No signal dimension means integrated ZLP intensity
         is provided. Signal dimension 1 means the ZLP model is provided and
         will be ingrated using `zlp.integrate1D`.
        t : {number, ndarray, signal}
         Thickess value or values provided in an array (signal) with correct
         (navigation) dimensions, as explained above for zlp.
        output : string, one of the following; {'full', 'diff'}
         Only used if ddcs == None. If output is 'full', a list of 3 signals is
         returned corresponding to the relativistic bulk+surf, relativistic bulk
         and semi-relativistic bulk DDCS obtained in get_relativistic_ddcs. Use
         'diff' to return the difference between the relativistic bulk + surface
         and the semi-relativistic bulk.
        ddcs : {Signal2D, list of}
         Provide your own DDCS, let your imagination fly...
        method : string, one of the following {'simpson', 'cubature',
                                               'lse', 'quadrature'}
         Select integration method to use. The available methods are; 'simpson',
         for Simpson-rule; 'cubature' and 'quadrature', for multi- or single-
         dimensional Gaussian quadrature; and, 'lse' for the log-sum-exp trick.
         Simpson-rule integration is the recommended default.
        Ncpu : int
         For quad integration.
        show_progressbar : {None, bool}
         Set to True to use display progressbar.

        *args, **kwargs passed to `self.get_relativistic_ddcs`

        Returns
        -------
        dcs : {Signal1D, list of}
         Depending on parameters a single or a list of signal1D objects.
        '''
        # TODO: self is single signal and zlp or t multi. Repeat accordingly...
        # recognize zlp spectrum or intensity
        if isinstance(zlp, hs.signals.Signal1D):
            if (zlp.axes_manager.signal_dimension == 1) and (
                zlp.axes_manager.navigation_shape ==
                 self.axes_manager.navigation_shape):
                axis = self.axes_manager.signal_axes[0]
                zlp = zlp.integrate1D(axis.index_in_axes_manager)
        zlp = self._check_adapt_map_input(zlp)

        thk = self._check_adapt_map_input(t)
        # TODO: also alert if something is wrong with zlp
        if isinstance(thk, ValueError):
            thk.args = (thk.args[0]+'t',)
            raise thk

        ddcs_list = []
        if ddcs is None:
            Ptot, Pvol, Pelf = self.get_relativistic_ddcs(t=thk,
                                                          *args, **kwargs)
            if output is 'full':
                ddcs_list = [Ptot, Pvol, Pelf]
            if output is 'diff':
                ddcs_list = [Ptot - Pelf,]
        elif isinstance(ddcs, hs.signals.BaseSignal):
            ddcs_list = [ddcs,]
        elif type(ddcs) is list:
            for item in ddcs:
                if not isinstance(item, hs.signals.BaseSignal):
                    raise ValueError('ddcs input not understood,'
                                     'should be signal or list of signals')
                ddcs_list += [item.deepcopy(),]
        else:
            raise ValueError('ddcs input not understood,'
                             'should be signal or list of signals')

        ret = []
        for p in ddcs_list:
            eax = p.axes_manager.signal_axes[0]
            energies = eax.axis.copy()

            aax = p.axes_manager.signal_axes[1]
            angles = aax.axis.copy()
            if aax.units.startswith('log10'):
                angles = 10**angles
            elif aax.units.startswith('log'):
                angles = np.exp(angles)

            p.data = np.nan_to_num(p.data)
            p.axes_manager[-1].axis = angles

            theta = np.repeat(angles[:, None], len(energies), 1)
            sinus = np.sin(theta)
            foo = lambda pdata: pdata*sinus
            p.map(foo, inplace=True, show_progressbar=show_progressbar)

            # Switch depending on the preferred integration method.
            if method is 'simpson':
                spc = p.integrate_simpson(-1)

            elif method is 'cubature':
                spc = p.map(interp_cuba,
                            xaxis=angles,
                            yaxis=energies,
                            inplace=False,
                            show_progressbar=show_progressbar)

            elif method is 'lse':
                spc = p.map(log_sum_exp,
                            xaxis=angles,
                            yaxis=energies,
                            inplace=False,
                            show_progressbar=show_progressbar)

            elif method is 'quadrature':
                # transpose and, if needed, unfold data (creating a view)
                Nd = p.axes_manager.navigation_dimension
                pdata = np.moveaxis(p.data, Nd, 0).reshape(len(angles),-1)
                # integrate using the wrapper
                pshape = p.axes_manager.navigation_shape[::-1] + (eax.size,)
                dummy_eax = np.arange(np.prod(pshape))*(eax.scale) + eax.offset
                spc = p.isig[:,0].deepcopy()
                quadw = interp_quad(angles, dummy_eax, pdata)
                spc.data = quadw.integrate_y(Ncpu=Ncpu).reshape(pshape)

            else:
                raise AttributeError('Parameter method not recognized. See docs'
                                     'for a list of available methods.')

            from eels import ModifiedEELS
            spc = ModifiedEELS(spc)

            spc = spc * zlp if (len(spc) != 1) else (spc * zlp.data[0])
            spc.data *= 2. * np.pi * eax.scale

            ret += [ModifiedEELS(spc),]

            # logsumexp integration
            # ... This also works, but is somehow slower.
            # ... However, under some circumstances, precission improves.
            #lsum, lsig = logsumexp(np.log(p.data), 2, b=angles[:,None], return_sign=True)
            #angles_log = np.log(angles)
            #logscale = angles_log[1] - angles_log[0]
            #ldata = np.exp(lsum*lsig) * logscale * 2. * np.pi

            #logse = p.deepcopy()
            #logse.data = ldata
            #logse._remove_axis(3)
            #logse = logse * zlp_intensity

            #ret += [logse,]

        if len(ret) == 1:
            return ret[0]
        else:
            return ret
