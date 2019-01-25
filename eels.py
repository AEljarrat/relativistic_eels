import numpy as np

from numbers import Number
from scipy import constants

# hyperspy dependency
import hyperspy.api as hs

from hyperspy.external.progressbar import progressbar
from hyperspy.defaults_parser import preferences

# own dependency
from signals import SignalMixin
from dielectric import ModifiedCDF

class ModifiedEELS(hs.signals.EELSSpectrum, SignalMixin):

    def __init__(self, *args, **kwargs):
        """
        Modified hypespy EELS signal. Input can be array or EELSSpectrum type.
        In the last case, additional *args and **kwargs are discarded.
        """
        if len(args) > 0 and isinstance(args[0], hs.signals.EELSSpectrum):
            # Pretend it is a hs signal, copy axes and metadata
            sdict = args[0]._to_dictionary()
            hs.signals.EELSSpectrum.__init__(self, **sdict)
        else:
            hs.signals.EELSSpectrum.__init__(self, *args, **kwargs)

    def get_effective_collection_angle(self):
        """
        Calculates the effective collection angle for the whole energy axis.
        The beam energy, convergence and collection angles need to be set in the
        metadata.

        Returns
        -------
        bstar : Signal1D
            Contains the effective collection energy for each energy-loss in
            mrad. It is a signal with the same length as the signal dimension.

        References
        ----------
        The calculation is done following Egerton (3rd edition) page 276.
        """
        try:
            e0 = self.metadata.Acquisition_instrument.TEM.beam_energy
        except BaseException:
            raise AttributeError("Please define the beam energy."
                                 "You can do this e.g. by using the "
                                 "set_microscope_parameters method")
        try:
            alpha = self.metadata.Acquisition_instrument.TEM.convergence_angle
        except BaseException:
            raise AttributeError("Please define the convergence semi-angle. "
                                 "You can do this e.g. by using the "
                                 "set_microscope_parameters method")
        try:
            beta = self.metadata.Acquisition_instrument.TEM.Detector.\
                EELS.collection_angle
        except BaseException:
            raise AttributeError("Please define the collection semi-angle. "
                                 "You can do this e.g. by using the "
                                 "set_microscope_parameters method")

        energy = self.axes_manager.signal_axes[0].axis

        tgt = e0 * (1.+e0/1022.) / (1.+e0/511.)
        thetae = (energy+1e-6) / tgt
        # A2,B2,T2 ARE SQUARES OF ANGLES IN RADIANS**2
        a2 = alpha * alpha * 1e-6 + 1e-10
        b2 = beta * beta * 1e-6
        t2 = thetae * thetae * 1e-6
        eta1 = np.sqrt((a2+b2+t2)**2 - 4.*a2*b2) - a2 - b2 - t2
        eta2 = 2.*b2 * np.log(0.5/t2*(np.sqrt((a2+t2-b2)**2 + 4.*b2*t2) \
                                                                + a2 + t2 - b2))
        eta3 = 2.*a2 * np.log(0.5/t2*(np.sqrt((b2+t2-a2)**2 + 4.*a2*t2) \
                                                                + b2 + t2 - a2))
        eta = (eta1+eta2+eta3) / a2 / np.log(4./t2)
        f1 = (eta1+eta2+eta3) / 2. / a2 / np.log(1.+b2/t2)
        f2 = f1
        if (alpha/beta > 1.):
            f2 = f1 * a2/b2

        bstar = thetae * np.sqrt(np.exp(f2*np.log(1.+b2/t2))-1.)

        bstar = self._get_signal_signal(bstar)
        bstar.set_signal_type('Signal1D')
        return bstar

    def model_zero_loss_peak_mirror(self, hanning=True):
        '''
        Model the zero-loss peak as a mirror using the negative energy-losses.
        This model only makes sense if the ZLP is symmetric around zero.

        Parameters
        ----------
        hanning : bool
         Optionally apply a Hanning taper to the border, so that they intensity
         decays smoothily to 0.

        Returns
        -------
        zlp : EELSSignal
         Mirrored ZLP model.
        '''
        zlp = self.deepcopy()
        eax = zlp.axes_manager.signal_axes[0]
        if eax.high_index*0.5 > eax.value2index(0.):
            eaxshift = - eax.scale*0.5
        elif eax.high_index*0.5 < eax.value2index(0.):
            eaxshift = eax.scale*0.5
        else:
            eaxshift = 0

        eax.offset = eax.offset + eaxshift

        plz = zlp.deepcopy().isig[0.::-1]
        xae = plz.axes_manager.signal_axes[0]
        xae.scale *= -1
        xae.offset *= -1
        plz.get_dimensions_from_data()
        Ea = xae.low_value
        Eb = xae.high_value
        Ec = eax.high_value

        i1 = eax.value2index(Ea)
        if Eb<Ec:
            i2 = eax.value2index(Eb)
            i3 = -1
        elif Eb>Ec:
            i2 = eax.high_index
            i3 = i2-xae.high_index-1
        else:
            i2 = eax.high_index
            i3 = -1

        zlp.data[(slice(None),)*eax.index_in_array+(slice(i1, i2), Ellipsis)]= \
         plz.data[(slice(None),)*xae.index_in_array+(slice(0, i3), Ellipsis)]

        zlp.data[(slice(None),)*eax.index_in_array + \
                 (slice(i2, None), Ellipsis)] = 0.

        eax = zlp.axes_manager.signal_axes[0]
        eax.offset = eax.offset-eaxshift

        if hanning:
            zlp_h = zlp.isig[:Eb-eaxshift]
            zlp_h.hanning_taper('both')

        return zlp

    def model_zero_loss_peak_tail(self, signal_range, show_progressbar=None,
                                  *args, **kwargs):
        '''
        Model the zero-loss peak tail using a (power-law) model fit. The fit
        window is set using the signal_range tuple in axis units (not indices).
        Spectral intensity at energies above the signal_range is substituted by
        the model tail. The fit is performed using `remove_background`,
        *args and **kwargs are passed to this method.

        Parameters
        ----------
        signal_range : tuple
         Initial and final position of the fit window. Given in axis units. The
         components can be single or multidimensional. If multidimensional, an
         array or a signal with the same dimensions as the navigation dimension
         should be used.

        Returns
        -------
        zlp : EELSSignal
         Modeled tail ZLP model.

        Examples
        --------
        >>> s = hs.load('some_eels.h5')
        >>> zlp = s.model_zero_loss_peak_tail((0.5, 1.),
        ...                                   fast=False,
        ...                                   show_progressbar=False)
        '''
        self._check_signal_dimension_equals_one()
        if not isinstance(signal_range, tuple):
            raise AttributeError('signal_range not recognized:'
                                 'must be a tuple!')

        if len(signal_range) != 2:
            raise AttributeError('signal_range not recognized,'
                                 'must be len = 2')

        axis = self.axes_manager.signal_axes[0]

        if isinstance(signal_range[0], Number) and (
           isinstance(signal_range[1], Number)):
           zlp = self - self.remove_background(signal_range,
                                              show_progressbar=show_progressbar,
                                              *args, **kwargs)
           I2 = axis.value2index(signal_range[1])
           ids = (slice(None),)*axis.index_in_array + (slice(None, I2), Ellipsis)
           zlp.data[ids] = self.data[ids]
           return zlp

        if isinstance(signal_range[0], (np.ndarray, hs.signals.BaseSignal)) or (
           isinstance(signal_range[1], (np.ndarray, hs.signals.BaseSignal))):
           signal_range_ini = self._check_adapt_map_input(signal_range[0])
           signal_range_fin = self._check_adapt_map_input(signal_range[1])
           for name in ['signal_range_ini', 'signal_range_fin']:
               parameter = eval(name)
               if isinstance(parameter, ValueError):
                   parameter.args = (parameter.args[0]+name,)
                   raise parameter
           zlp = self.deepcopy()

           for si in progressbar(self, disable=not show_progressbar):
               indices = self.axes_manager.indices
               E1 = signal_range_ini.inav[indices].data[0]
               E2 = signal_range_fin.inav[indices].data[0]
               ri = si.remove_background((E1, E2), show_progressbar=False,
                                         *args, **kwargs)
               I2 = axis.value2index(E2)
               ids  = (*indices, slice(I2, None), Ellipsis)
               zlp.data[ids] = si.data[I2:] - ri.data[I2:]
           return zlp

    def _get_ZeroLossPeak_model(self,
                                background=False,
                                compression=False,
                                factor = 0.99,
                                width = 0.5,
                                N = 4):
        """
        Create a zero-loss peak model using the ZeroLossPeak custom component.
        The model contains a Voigt peak and an optional intensity offset to
        simulate the background. Additionally, a compression filter can be
        calculated to give more importance to the zero-loss tails. This option
        only works when the model is used in conjuntion with
        model_zero_loss_peak.

        Parameters
        ----------
        background : bool
         Adds an intensity offset to the model to simulate background intensity.
        compression : bool
         Sets a compression filter to reduce the intensity of the ZLP.
        factor, width, N : floats
         These arguments are passed to the use_compression method of the
         ZeroLossPeak component. More information in the documentation therein.

        Returns
        -------
        m : hyperspy model
         Containing a ZeroLossPeak component, initialized to default values.
        """
        m = self.create_model(auto_background = False,
                              auto_add_edges  = False)
        from components import ZeroLossPeak
        ZLP = ZeroLossPeak()
        m.append(ZLP)
        m.set_parameters_value('FWHM', 0.2)
        m.set_parameters_value('gamma', 0.05)
        m.set_parameters_value('area', 1.)
        m.set_parameters_value('centre', 0.)
        m.set_parameters_value('non_isochromaticity', 0.1)
        m.set_parameters_free(['ZeroLossPeak',], ['non_isochromaticity',])

        # TODO: improve estimate area
        ZLP.area.map['values'] = self.estimate_elastic_scattering_intensity(1.)

        if compression:
            # sets a compression filter for the most intense part of the ZLP
            ZLP.use_compression(factor=factor, width=width, N=N)

        if background:

            # background adds an intensity offset to the ZLP model
            bck = ZLP.background

            # heuristics to predict offset
            offset_max = self.data[..., :10].mean(-1) / m.axis.scale
            ZLP.background.map['values'] = offset_max * 0.1
            ZLP.background.map['is_set'] = True

            # TODO: the fit bounding is not working?
            m.set_boundaries()
            bck.ext_force_positive = True
            bck.ext_bounded = True
            bck._set_bmax(offset_max.max())
            bck._set_bmin(0.)
            m.set_parameters_free(['ZeroLossPeak',], ['background',])

        return m

    def model_zero_loss_peak(self,
                             threshold=None,
                             model=None,
                             energy_window='auto',
                             return_model=False,
                             show_progressbar=None,
                             *args, **kwargs):
        """
        Flexible tool to model the zero-loss peak using a Voigt function and fit
        the spectrum up to a threshold energy. The result is given as a signal.
        The zero-loss peak model is created in a symmetric energy window.
        This window should be large enough so that the tails decay smoothly to
        zero, and the model can be used for (de)convolution purposes. Background
        can be alternatively added to model dark current counting error.

        Parameters
        ----------
        threshold : {None, int, float}
         Positive truncation energy to model the shape of the zero-loss peak,
         especified in energy/index units by passing float/int. By default, a
         value equal to the negative energy-loss limit is selected.
        model : hyperspy model
         If provided, this model is used to fit the ZLP in the provided signal.
         By default a model containing the ZeroLossPeak component (with a
         modified Voigt peak) is used by default. Note that the model.signal
         parameter is removed after the model is created, to save memory.
        energy_window : {'auto', float}
         Positive limit of the zero-centered symmetric energy window in which
         the resulting signal is created. Set to 'auto' to use the full range.
        return_model : bool
         Returns the fitted model instead of a signal.
        show_progressbar : {None, bool}
         Progressbar choice.
        *args, **kwargs, passed to _get_ZeroLossPeak_model function.

        Returns
        -------
        zlp : ModifiedEELS || model
         A zero-loss peak model with the same dimensions as the input signal.
        """

        self._check_signal_dimension_equals_one()
        axis  = self.axes_manager.signal_axes[-1]
        zlp_ini = float( axis.low_value )
        zlp_fin = float( axis.high_value + axis.scale )
        assert ( zlp_ini < 0. ) and ( zlp_fin > 0. )

        # set fit range
        if threshold is None:
            fit_range = (zlp_ini, -zlp_ini)
        else:
            fit_range = (zlp_ini, threshold)

        # set window range
        if energy_window is 'auto':
            win_range = (-zlp_fin, zlp_fin)
        else:
            win_range = (-energy_window, energy_window)

        # Extract ZLP data using expand and crop technique
        z = self.expand_signal1D(*win_range, inplace=False)
        z.crop_signal1D(*win_range)

        # Process model data
        if model is None:
            # ZeroLossPeak model
            model = z._get_ZeroLossPeak_model(*args, **kwargs)
        else:
            # user input model
            compdict = model.as_dictionary(fullcopy=True)
            model = z.create_model(auto_background = False,
                                   auto_add_edges  = False,
                                   dictionary      = compdict)
            # TODO: this is a workaround
            model.channel_switches = np.array([True] * len(model.axis.axis))

        # Apply compression if necessary
        cfunc = model.components.ZeroLossPeak.compression
        if cfunc is not None:
            model.signal.data *= cfunc(model.axis.axis)

        # Fit the chosen model to the ZLP data
        model.set_signal_range(*fit_range)
        model.multifit(show_progressbar=show_progressbar)

        if return_model:
            return model

        else:
            # Create zlp model
            model.reset_signal_range()
            zlp = model.as_signal(show_progressbar=show_progressbar)
            zlp = ModifiedEELS(zlp)

            if cfunc is not None:

                # Remove compression from ZLP model
                carr = cfunc(zlp.axes_manager[-1].axis)
                zlp = zlp / carr

                # correct negative eloss tail
                ndim = self.axes_manager.navigation_dimension
                de = axis.scale
                dtail = self.isig[zlp_ini] - zlp.isig[zlp_ini]
                zslice = zlp.isig[:zlp_ini+de]
                zslice.data += dtail.data[..., None] if ndim > 0 else dtail.data

                # Use experimental data within compression limits
                clim = zlp.axes_manager[-1].axis[(carr - 0.95)<0.][-1]
                zslice = zlp.isig[zlp_ini:clim]
                sslice = self.isig[zlp_ini:clim]
                zslice.data[:] = sslice.data[:]

                # correct positive eloss tail
                dtail = self.isig[clim] - zlp.isig[clim]
                zslice = zlp.isig[clim:]
                zslice.data += dtail.data[..., None] if ndim > 0 else dtail.data

                # finish touches
                zlp.remove_negative_intensity(inplace=True)
                #zlp.hanning_taper('both')
                ncl = int((zlp_fin - zlp_ini)//(de*2))
                ncr = int((zlp_fin - clim)//(de*2))
                zlp.hanning_taper(side='left', channels=ncl)
                zlp.hanning_taper(side='right', channels=ncr)

            return zlp

    def fourier_exp_convolution(self, zlp):
        '''
        Poisson statistics to simulate the effect of plural inelastic scattering
        using the exponential formula from Egerton (see eq. 4.8, pp 233); e.g.
        starting from an input single-scattering distribution (SSD) and zero-
        loss peak (ZLP) models, simulate the recorded EELS spectrum. This is
        implemented using FFTs as the product of the exponential of the SSD and
        the ZLP.

        Note that because FEC uses FFT, care should be taken to ensure the input
        SSD and ZLP meet periodic boundary constraints, for instance by allowing
        the intensity to decay smoothly to zero at both ends of the energy axis.
        The algorithm also assumes both signals have the same scale, but not
        necessarily the same number of channels or offset.

        Parameters
        ----------
        zlp : signal1D
         A signal containing the ZLP, must have the same navigation dimensions
         as the input or none for a single zlp model. In this last case, the zlp
         is broadcasted to convolve a single model to the whole input.

        Returns
        -------
        eels : signal1D
         A signal that includes the ZLP and a simulated plural scattering
         distribution corresponding to the inputs. Fourier-log deconvolution can
         be applied to this signal to retrieve the input (see Example below).

        Examples
        --------
         >>> eels = ssd.fourier_exp_convolution(zlp)
         >>> ssd_estimate = eels.fourier_log_deconvolution(zlp)

        References
        ----------
        R.F. Egerton "EELS in the electron microscope" 2011, Springer.
        '''
        # The output signal should look like the input signal
        eels = self.deepcopy()

        axis_ssd = self.axes_manager.signal_axes[0]
        axis_zlp = zlp.axes_manager.signal_axes[0]

        # FFTs are padded up to double the input signal size
        tsize = (2*axis_ssd.size)

        # to calculate the time-shift
        tshift = lambda axis : np.exp(-2j*np.pi*axis.offset / \
                                      axis.scale*np.fft.fftfreq(tsize))

        # Calculate the inverset ZLP integral
        # and take care of broadcasting for single zlp input
        I0 = zlp.integrate1D(-1)**-1
        I0 = I0.data[..., None] if (len(I0) != 1) else float(I0.data)

        # The FFTs are done with 2 times energy-loss axis size
        kwfft = {'n':tsize, 'axis':-1}
        eels.data  = tshift(axis_ssd)*np.fft.fft(self.data, **kwfft)
        eels.data  = np.exp(I0*eels.data)
        eels.data *= tshift(axis_zlp)*np.fft.fft(zlp.data, **kwfft)
        eels.data  = np.fft.ifft(eels.data, **kwfft).real
        eels.get_dimensions_from_data()

        eels.data = np.roll(eels.data, -int(axis_ssd.offset/axis_ssd.scale), -1)
        eels.crop_signal1D(None, axis_ssd.size)
        return eels

    def fourier_log_deconvolution(self, zlp):
        '''
        Poisson statistics to remove the effect of plural inelastic scattering
        using the logarithmic formula from Egerton (see eq. 4.11, pp 234); e.g.
        starting from an input energy-loss spectrum (EELS) and zero-loss peak
        (ZLP) model, extract the SSD spectrum. This is implemented using FFTs as
        the ratio of the logarithm of the EELS and the ZLP.

        Note that because FLD uses FFT, care should be taken to ensure the input
        EELS + ZLP meet periodic boundary constraints, for instance by allowing
        the intensity to decay smoothly to zero at both ends of the energy axis.
        The algorithm also assumes both signals have the same scale, but not
        necessarily the same number of channels or offset.

        Parameters
        ----------
        zlp : signal1D
         A signal containing the ZLP, must have the same navigation dimensions
         as the input or none for a single zlp model. In this last case, the zlp
         is broadcasted to deconvolve a single model from the whole input.

        Returns
        -------
        ssd : signal1D
         An extracted signal, removing the ZLP and plural scattering
         distribution from the input EELS. Fourier-log deconvolution can be
         applied to this signal to retrieve the input (see Example below).

        Examples
        --------
         >>> eels = ssd.fourier_exp_convolution(zlp)
         >>> ssd_estimate = eels.fourier_log_deconvolution(zlp)

        References
        ----------
        R.F. Egerton "EELS in the electron microscope" 2011, Springer.
        '''
        # The output signal should look like the input signal
        ssd = self.deepcopy()

        axis_spc = self.axes_manager.signal_axes[0]
        axis_zlp = zlp.axes_manager.signal_axes[0]

        # FFTs are padded up to double the input signal size
        tsize = (2*axis_spc.size)

        # to calculate the time-shift
        tshift = lambda axis : np.exp(-2j*np.pi*axis.offset / \
                                      axis.scale*np.fft.fftfreq(tsize))

        # The FFTs are done with 2 times energy-loss axis size
        kwfft = {'n':tsize, 'axis':-1}
        ssd.data  = tshift(axis_spc) * np.fft.fft(self.data, **kwfft)
        z = tshift(axis_zlp) * np.fft.fft(zlp.data, **kwfft)
        ssd.data /= z
        ssd.data  = z * np.nan_to_num(np.log(ssd.data))
        ssd.data  = np.fft.ifft(ssd.data, **kwfft).real

        ssd.data = np.roll(ssd.data, -int(axis_spc.offset/axis_spc.scale), -1)
        ssd.crop_signal1D(None, axis_spc.size)
        return ssd

    def power_law_extrapolation_until(self, window_size=20, total_size=1024,
                                      hanning=True, *args, **kwargs):
        '''
        Extend the spectrum using  `power_law_extrapolation` until the resulting
        spectrum has the given total number of pixels.

        Parameters
        ----------
        window_size : {int, float}
         Window size. Alternatively, this parameter can be a float specifying
         the size in energy-loss units.
        total_size : {int, float}
         Total number of pixels. Alternatively, this parameter can be a float
         specifying the size in signal axis units.

        Returns
        -------
        spc : EELSSpectrum
         The extended spectrum.

        '''
        axis = self.axes_manager.signal_axes[0]

        if type(total_size) is float:
            offset = axis.offset - axis.scale
            total_size = int((total_size-offset)//axis.scale)

        if type(window_size) is float:
            window_size = int(window_size//axis.scale)

        extrapolation_size = total_size-axis.size
        if extrapolation_size < 0:
            raise AttributeError("total_size is less than spectral axis size")

        spc = self.power_law_extrapolation(window_size,
                                           extrapolation_size,
                                           *args,
                                           **kwargs)

        if hanning:
            spc.hanning_taper('left', channels=10)
            ncl = int(extrapolation_size//2)
            spc.hanning_taper(side='right', channels=ncl)

        return spc

    def normalize_bulk_inelastic_scattering(self, zlp=1., n=None, t=None):
        """
        Normalize a the bulk inelastic scattering distribution using Ritchie's
        formula. This formula does not take into account radiative or surface
        modes. The normalization can be performed using the refractive index or
        the thickness as input parameters. Providing one of these selects the
        normalization procedure.

        Parameters
        ----------
        zlp: {None, number, ndarray, Signal}
         Optionally provide the ZLP intensity, that is in principle needed for
         normalization of the inelastic scattering distribution and to calculate
         the absolute thickness value.
        n: {None, number, ndarray, Signal}
         Optionally provide the refractive index of the medium. If provided, the
         thickness parameter `t` should be None, or else an error is raised.
        t: {None, number, ndarray, Signal}
         Optionally provide the sample thickness, in nm. If provided, the
         refractive index parameter `n` should be None, or else an error is
         raised.

        Returns
        -------
        elf: ModifiedEELS
         The imaginary part of the inverse dielectric function, also known as
         energy-loss function.
        t : Signal
         The thickness, estimated from the refractive index in case this was
         provided. In case the thickness was already provided, we have made sure
         to adapt the navigation dimensions of the input signals.
        """

        # create energy-loss axis
        axis   = self.axes_manager.signal_axes[0]
        energy = axis.axis.copy()

        # recognize zlp spectrum or intensity
        if isinstance(zlp, hs.signals.Signal1D):
            if (zlp.axes_manager.signal_dimension == 1) and (
                zlp.axes_manager.navigation_shape ==
                 self.axes_manager.navigation_shape):
                Izlp = zlp.integrate1D(axis.index_in_axes_manager)

        # the input parameters should agree with the navigation dimensions
        Izlp = self._check_adapt_map_input(zlp)
        n = self._check_adapt_map_input(n)
        t = self._check_adapt_map_input(t)
        for name in ['n', 't', 'Izlp']:
            parameter = eval(name)
            if isinstance(parameter, ValueError):
                parameter.args = (parameter.args[0]+name,)
                raise parameter

        # select refractive or thickness loop
        if n is None and t is None:
            raise ValueError('Thickness and refractive index undefined.'
                             'Please provide one of them.')
        elif n is not None and t is not None:
            raise ValueError('Thickness and refractive index both defined.'
                             'Please provide only one of them.')
        elif n is not None:
            sum_rule_norm = True
            if Izlp is not None:
                t = self._get_navigation_signal().T
        elif t is not None:
            sum_rule_norm = False
            if Izlp is None:
                raise ValueError('Zero-loss intensity is needed for thickness '
                                 'normalization. Provide also parameter zlp')

        # Constants and units, electron mass, beam energy and collection angle
        me = constants.value(
            'electron mass energy equivalent in MeV') * 1e3  # keV
        try:
            e0 = self.metadata.Acquisition_instrument.TEM.beam_energy
        except BaseException:
            raise AttributeError("Please define the beam energy."
                                 "You can do this e.g. by using the "
                                 "set_microscope_parameters method")
        try:
            beta = self.metadata.Acquisition_instrument.TEM.Detector.\
                EELS.collection_angle
        except BaseException:
            raise AttributeError("Please define the collection semi-angle. "
                                 "You can do this e.g. by using the "
                                 "set_microscope_parameters method")

        ## Kinetic definitions
        ke  = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2
        tgt = e0 * (2 * me + e0) / (me + e0)

        # output
        elf = self.deepcopy()

        # Calculation of the ELF by normalization of the SSD
        elf.data = self.data / \
                           (np.log(1 + (beta * tgt / energy) ** 2) * axis.scale)

        if sum_rule_norm:
            # normalize using the refractive index.
            K = (elf.data/energy).sum(axis=axis.index_in_array)*axis.scale
            K = (K / (np.pi / 2) / (1 - 1. / n.data ** 2))
            # Update the thickness if refractive index is present
            te = (332.5 * K * ke / Izlp.data)
            t.data = te.squeeze()
        else:
            # normalize using the thickness
            K = t.data * Izlp.data / (332.5 * ke)

        elf.data = elf.data / K[..., None] if len(self) != 1 else elf.data / K
        return elf, t

    def kramers_kronig_transform(self, invert=True):
        """
        From a normalized energy-loss function, calculate the corresponding
        Kramers-Kronig transform using FFTs. In this algorithm, the data is
        padded with zeros up to double its size to avoid the wrap-around
        problem. The energy axis is taken to be the last one in the dataset.
        Additional functionality is available with the parameters.

        Parameters
        ----------
        invert : bool
         If True, the input is taken to be the imaginary part of the inverse
         dielectric function, e.g. the energy-loss function obtained from a
         normalized semi-classical bulk inelastic distribution. If False, the
         input is interpreted as the imaginary part of the dielectric function.

        Returns
        -------
        cdf : ModifiedCDF
         The complex dielectric function including with the input in the
         imaginary part and the corresponding Kramers-Kronig Transform in the
         real.
        """
        imdata = self.data
        axis   = self.axes_manager.signal_axes[0]
        esize  = axis.size
        dsize  = 2 * axis.size
        ids    = self.axes_manager._get_data_slice(
                          [(axis.index_in_array, slice(None, axis.size)), ])
        q      = - 2 * np.fft.fft(imdata, dsize, -1).imag / dsize
        q[ids] *= -1.
        q      = np.fft.fft(q, axis=-1)
        # prepare the output dielectric function
        cdf = self._deepcopy_with_new_data( 1. + 1j*self.data )
        cdf = ModifiedCDF(cdf)
        if invert:
            cdf.data += q[ids].real
            cdf.data /= ( cdf.data.real**2. + cdf.data.imag**2. )
        else:
            cdf.data -= q[ids].real
        return cdf

    def get_single_scattering_distribution(self, zlp, kwpad=None):
        """
        Extract the SSD by deconvolution of a zlp model from the EELS spectrum.

        Parameters
        ----------
        zlp : hyperspy model
         Input for `~.model_zero_loss_peak`
        kwpad : {None, dictionary}
         Optionally containing the input parameters for the pad. This is
         performed using the `~.power_law_extrapolation_until` method. More
         information in the docs therein.

        Returns
        -------
        ssd : SingleScatteringDistribution
         The deconvolved single scattering distribution signal.
        z : ModifiedEELS
         The deconvolved zero-loss peak signal.
        """

        zlp_threshold = float(zlp.axis.axis[zlp.channel_switches][-1])
        axis = self.axes_manager[-1]
        ssd_range = (float(axis.scale), float(axis.high_value+axis.scale))

        # Update ZLP model with appropriate size (expand and crop technique)
        z = self.model_zero_loss_peak(threshold = zlp_threshold,
                                     model     = zlp,
                                     show_progressbar = False)
        ssd = self.deepcopy()
        if kwpad is not None:
            # Pad the EELS spectra
            ssd = ssd.power_law_extrapolation_until(**kwpad)

        # extract SSD using the Fourier-log deconvolution
        ssd = ssd.fourier_log_deconvolution(z)
        ssd.remove_negative_intensity()
        if kwpad is not None:
            ssd.crop_signal1D(*ssd_range)
        ssd = SingleScatteringDistribution(ssd)
        return ssd, z

    def obtain_dielectric_function(self, z, n=None, t=None, kwpad=None,
                                   background=None):
        """
        Calculates the dielectric function for the provided ZLP and
        refractive index.

        Parameters
        ----------
        z : hyperspy Signal
         Containing the zero-loss peak, over an energy range as broad as
         possible.
        n, t : float, hyperspy Signal
         Optionally provide either the refractive index or the thickness,
         that are used for normalization of the SSD. In case both are
         provided, ValueError is raised.
        kwpad: dictionary
         Optionally containing the input parameters for the pad. This is
         performed using the `~.power_law_extrapolation_until` method. More
         information in the docs therein.
        background : {None, float, hyperspy Signal}
         Optionally provide an intensity offset that is removed from the spectra
         (also ZLP) prior to the other calculations.

        Returns
        -------
        eps : ModifiedCDF
         The dielectric function.
        thickness: hyperspy Signals
         The corresponding thickness.
        """

        axis = self.axes_manager.signal_axes[0]
        ssd_range = (float(axis.scale), float(axis.high_value+axis.scale))

        if background is not None:
            Izlp = (z - background).integrate1D(-1)
        else:
            Izlp = z.integrate1D(-1)

        if kwpad is not None:
            # Pad the EELS spectra
            ssd_p = self.power_law_extrapolation_until(**kwpad)

        elf, thickness = ssd_p.normalize_bulk_inelastic_scattering(Izlp,
                                                                   n=n,
                                                                   t=t)
        eps = elf.kramers_kronig_transform(invert=True)
        eps.crop(-1, *ssd_range)

        return eps, thickness


    def relativistic_kramers_kronig(self,
                                    zlp=None,
                                    n=None,
                                    t=None,
                                    delta=0.9,
                                    fsmooth=None,
                                    iterations=20,
                                    chi2_target=1e-4,
                                    average=False,
                                    full_output=True,
                                    show_progressbar=None,
                                    *args,
                                    **kwargs):
        r"""Calculate the complex dielectric function from a single scattering
        distribution (SSD) using the Kramers-Kronig relations and a relativistic
        correction for thin slab geometry.

        The input SSD should be and EELSSpectrum instance, containing only
        inelastic scattering information (elastic and plural scattering
        deconvolved). The dielectric information is obtained by normalization of
        the inelastic scattering using the elastic scattering intensity and
        either refractive index or thickness information.

        A full complex dielectric function (CDF) is obtained by Kramers-Kronig
        transform, solved using FFT as in `kramers_kronig_analysis`. This inital
        guess for the CDF is improved in an iterative loop, devised to
        approximately subtract the relativistic contribution supposing an
        unoxidized planar surface.

        The loop runs until a chi-square target has been achieved or for a
        maximum number of iterations. This behavior can be modified using the
        parameters below. This method does not account for instrumental and
        finite-size effects.

        Note: either refractive index or thickness (`n` or `t`) are required.
        If both are None or if both are provided an exception is raised. Many
        input types are accepted for zlp, n and t parameters, which are parsed
        using `self._check_adapt_map_input`, see the documentation therein for
        more information.

        Parameters
        ----------
        zlp: {None, number, ndarray, Signal}
            ZLP intensity. It is optional (can be None) if t is given,
            full_output is False and no iterations are run. In any other case,
            the ZLP is required either to perform the normalization step,
            to calculate the thickness and/or to calculate the relativistic
            correction.
        n: {None, number, ndarray, Signal}
            The medium refractive index. Used for normalization of the
            SSD to obtain the energy loss function. If given the thickness
            is estimated and returned. It is only required when `t` is None.
        t: {None, number, ndarray, Signal}
            The sample thickness in nm. Used for normalization of the
            SSD to obtain the energy loss function. It is only required when
            `n` is None.
        delta : {None, float}
            Optionally apply a fractional limit to the relativistic correction
            in order to improve stability. Can be None, if no limit is desired.
            A value of around 0.9 ensures the correction is never larger than
            the original EELS signal, producing a negative spectral region.
        fsmooth : {None, float}
            Optionally apply a gaussian filter to the relativistic correction
            in order to eliminate high-frequency noise. The cut-off is set in
            the energy-loss scale, e.g. fsmooth = 1.5 (eV).
        iterations: {None, int}
            Number of the iterations for the internal loop to remove the
            relativistic contribution. If None, the loop runs until a chi-square
            target has been achieved (see below).
        chi2_target : float
            The average chi-square test score is measured in each iteration, and
            the reconstruction loop terminates when the target score is reached.
            See `_chi2_score` for more information.
        average : bool
            If True, use the average of the obtained dielectric functions over
            the navigation dimensions to calculate the relativistic correction.
            False by default, should only be used when analyzing spectra from a
            homogenous sample, as only one dielectric function is retrieved.
            This switch has no effect if only one spectrum is being analyzed.
        full_output : bool
            If True, return a dictionary that contains the estimated
            thickness if `t` is None and the estimated relativistic correction
            if `iterations` > 1.

        Returns
        -------
        eps: DielectricFunction instance
            The complex dielectric function results,

                .. math::
                    \epsilon = \epsilon_1 + i*\epsilon_2,

            contained in an DielectricFunction instance.
        output: Dictionary (optional)
            A dictionary of optional outputs with the following keys:

            ``thickness``
                The estimated thickness in nm calculated by normalization of
                the corrected spectrum (only when `t` is None).

            ``relativistic correction``
               The estimated relativistic correction at the final iteration.

        Raises
        ------
        ValueError
            If both `n` and `t` are undefined (None).
        AttributeError
            If the beam_energy or the collection semi-angle are not defined in
            metadata.

        See also
        --------
        get_relativistic_spectrum, _check_adapt_map_input

        """
        # prepare data arrays
        if iterations == 1:
            # In this case s.data is not modified so there is no need to make
            # a deep copy.
            s = self.isig[0.:]
        else:
            s = self.isig[0.:].deepcopy()

        sorig = self.isig[0.:]

        # Avoid singularity at 0
        if s.axes_manager.signal_axes[0].axis[0] == 0:
            s = s.isig[1:]
            sorig = self.isig[1:]
        axis = s.axes_manager.signal_axes[0]
        eaxis = axis.axis.copy()

        # Constants and units, electron mass, beam energy and collection angle
        me = constants.value(
            'electron mass energy equivalent in MeV') * 1e3  # keV
        try:
            e0 = s.metadata.Acquisition_instrument.TEM.beam_energy
        except BaseException:
            raise AttributeError("Please define the beam energy."
                                 "You can do this e.g. by using the "
                                 "set_microscope_parameters method")
        try:
            beta = s.metadata.Acquisition_instrument.TEM.Detector.\
                EELS.collection_angle
        except BaseException:
            raise AttributeError("Please define the collection semi-angle. "
                                 "You can do this e.g. by using the "
                                 "set_microscope_parameters method")

        # Mapped parameters, zlp, n and t
        if isinstance(zlp, hs.signals.Signal1D):
            if (zlp.axes_manager.signal_dimension == 1) and (
                zlp.axes_manager.navigation_shape ==
                 self.axes_manager.navigation_shape):
                zlp = zlp.integrate1D(axis.index_in_axes_manager)
        elif zlp is None and (full_output or iterations > 1):
            raise AttributeError("Please define the zlp parameter when "
                                 "full output or iterations > 1 are selected.")
        zlp = self._check_adapt_map_input(zlp)
        n = self._check_adapt_map_input(n)
        t = self._check_adapt_map_input(t)
        for name in ['zlp', 'n', 't']:
            parameter = eval(name)
            if isinstance(parameter, ValueError):
                parameter.args = (parameter.args[0]+name,)
                raise parameter

        # select refractive or thickness loop
        if n is None and t is None:
            raise ValueError('Thickness and refractive index undefined.'
                             'Please provide one of them.')
        elif n is not None and t is not None:
            raise ValueError('Thickness and refractive index both defined.'
                             'Please provide only one of them.')
        elif n is not None:
            refractive_loop = True
            if (zlp is not None) and (full_output is True or iterations > 1):
                t = self._get_navigation_signal().T
        elif t is not None:
            refractive_loop = False
            if zlp is None:
                raise ValueError('Zero-loss intensity is needed for thickness '
                                 'normalization. Provide also parameter zlp')

        # Slicer to get the signal data from 0 to axis.size
        slicer = s.axes_manager._get_data_slice(
            [(axis.index_in_array, slice(None, axis.size)), ])

        # Kinetic definitions
        ke = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2
        tgt = e0 * (2 * me + e0) / (me + e0)
        rk0 = 2590 * (1 + e0 / me) * np.sqrt(2 * ke / me)

        # prepare the output dielectric function
        eps = s._deepcopy_with_new_data(np.zeros_like(s.data, np.complex128))
        eps.set_signal_type("DielectricFunction")
        eps.metadata.General.title = (self.metadata.General.title +
                                      'KKA dielectric function')
        if eps.tmp_parameters.has_item('filename'):
            eps.tmp_parameters.filename = (
                self.tmp_parameters.filename +
                '_CDF_after_Kramers_Kronig_transform')

        from dielectric import ModifiedCDF
        eps = ModifiedCDF(eps)
        eps_corr = eps.deepcopy()

        # progressbar support
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        pbar = progressbar(total = iterations,
                           desc = '1.00e+30',
                           disable=not show_progressbar)

        # initialize iteration control
        io = 0
        chi2 = chi2_target*1e3
        while (io < iterations) and (chi2 > chi2_target):
            # Calculation of the ELF by normalization of the SSD
            Im = s.data / (np.log(1 + (beta * tgt / eaxis) ** 2)) / axis.scale

            if refractive_loop:
                # normalize using the refractive index.
                K = (Im / eaxis).sum(axis=axis.index_in_array) * axis.scale
                K = (K / (np.pi / 2) / (1 - 1. / n.data ** 2))
                # Calculate the thickness only if possible and required
                if full_output or iterations > 1:
                    te = (332.5 * K * ke / zlp.data)
                    t.data = te.squeeze()
            else:
                # normalize using the thickness
                K = t.data * zlp.data / (332.5 * ke)
            Im = Im / K[..., None] if len(self) != 1 else Im / K

            # Kramers-Kronig transform
            esize = 2 * axis.size
            q = -2 * np.fft.fft(Im, esize, axis.index_in_array).imag / esize
            q[slicer] *= -1
            q = np.fft.fft(q, axis=axis.index_in_array)
            Re = q[slicer].real + 1
            epsabs = (Re ** 2 + Im ** 2)
            eps.data =  Re / epsabs + 1j* Im / epsabs
            del Im, Re, q, epsabs

            if average and (eps.axes_manager.navigation_dimension > 0):
                eps_corr.data[:] = eps.data.mean(
                                   eps.axes_manager.navigation_indices_in_array,
                                   keepdims=True)
            else:
                eps_corr.data = eps.data.copy()

            if full_output or iterations > 1:
                # Relativistic correction
                #  Calculates relativistic correction from the Kroeger equation
                #  The difference with the relativistic DCS is subtracted
                scorr = eps_corr.get_relativistic_spectrum(zlp=zlp, t=t,
                                                           output='diff',
                                                           *args, **kwargs)
                # Limit the fractional correction
                if delta is not None:
                    fcorr = np.clip(scorr.data / sorig.data, -delta, delta)
                    scorr.data = fcorr * sorig.data

                # smooth
                if fsmooth is not None:
                    scorr.gaussian_filter(fsmooth)

                # Apply correction
                s.data = sorig.data - scorr.data
                s.data[s.data < 0.] = 0.

                if io > 0:
                    #chi2 = ((scorr.data-smemory)**2/smemory**2).sum()
                    chi2 = smemory._chi2_score(scorr)
                    chi2str = '{:0.2e}'.format(chi2)
                    pbar.set_description(chi2str)
                smemory = scorr.deepcopy()
            io += 1
            pbar.update(1)

        pbar.close()

        if full_output:
            output = {}
            tstr = self.metadata.General.title
            if refractive_loop:
                t.metadata.General.title = tstr + ', r-KKA thickness'
                output['thickness'] = t
            scorr.metadata.General.title = tstr + ',  r-KKA correction'
            output['relativistic correction'] = scorr
            return eps, output
        else:
            return eps

    def relativistic_kramers_kronig_zlp(self,
                                        zlp=None,
                                        n=None,
                                        t=None,
                                        delta=0.9,
                                        iterations=20,
                                        chi2_target=1e-4,
                                        average=False,
                                        ssd_threshold=None,
                                        zlp_threshold=None,
                                        kwpad=None,
                                        show_progressbar=None,
                                        *args,
                                        **kwargs):
        r"""Calculate the complex dielectric function from a low-loss EELS
        dataset, using the the Kramers-Kronig relations and a relativistic
        correction for thin slab geometry.

        The input dataset should contain most of the elastic and inelastic
        scattering distributions. A zero-loss peak model is used to remove and
        include the elastic scattering distribution from the estimations. This
        process is mostly automated but can be controlled by providing a custom
        zero-loss peak model using the `zlp` parameter. Finer control is also
        possible, for more information see parameter list.

        Parameters
        ----------
        zlp: EELSModel
         ZLP model containing a ZeroLossPeak component to fit the zlp. It can be
         obtained by using the `model_zero_loss_peak` method.
        n: {None, number, ndarray, Signal}
         The medium refractive index. Used for normalization of the SSD to
         obtain the energy loss function. It is only required when `t` is None.
        t: {None, number, ndarray, Signal}
         The sample thickness in nm. Used for normalization of the SSD to obtain
         the energy loss function. It is only required when `n` is None.
        delta : {None, float}
         Optionally apply a fractional limit to the relativistic correction in
         order to improve stability. Can be None, if no limit is desired. A
         value of around 0.9 ensures the correction is never larger than the
         original EELS signal, producing a negative spectral region.
        iterations: {None, int}
         Number of the iterations for the internal loop to remove the
         relativistic contribution. If None, the loop runs until a chi-square
         target has been achieved (see below).
        chi2_target : float
         The average chi-square test score is measured in each iteration, and
         the reconstruction loop terminates when the target score is reached.
         See `_chi2_score` for more information.
        average : bool
         If True, use the average of the obtained dielectric functions over
         the navigation dimensions to calculate the relativistic correction.
         False by default, should only be used when analyzing spectra from a
         homogenous sample, as only one dielectric function is retrieved. This
         switch has no effect if only one spectrum is being analyzed.
        ssd_threshold : {None, float}
         Optionally, crop the single scattering distribution prior to
         normalization using this value as a lower boundary. By default, the
         full spectra starting from the first positive channel are used.
        zlp_threshold : {None, float}
         Optionally, provide the upper boundary for the ZLP fit. This fit is
         performed using the model_zero_loss_peak method. By default, the fit
         range is set simetrically around the ZLP using the first channel as
         lower boundary.
        kwpad : {None, dictionary}
         Optional paramater. A dictionary with key-word arguments for the
         `power_law_extrapolation` method. If provided, this method will be
         used to extend the high-energy tail of the spectra. In most cases, this
         padding is necessary to obtain correct Fourier transforms.

        Returns
        -------
        eps: ModifiedCDF
         The complex dielectric function resulting from the analysis.
        output: Dictionary
         A dictionary of outputs with the following keys:

            ``thickness``
                The estimated thickness in nm calculated by normalization of
                the corrected spectrum (only when `t` is None).

            ``relativistic correction``
               The estimated relativistic correction at the final iteration.

        Raises
        ------
        ValueError
            If both `n` and `t` are undefined (None).
        AttributeError
            If the beam_energy or the collection semi-angle are not defined in
            metadata.

        See also
        --------
        power_law_extrapolation_until, get_relativistic_spectrum,
        _check_adapt_map_input, normalize_bulk_inelastic_scattering,
        kramers_kronig_transform


        """

        # Calculate the energy ranges
        axis = self.axes_manager[-1]

        eel_range = [float(axis.low_value),
                     float(axis.high_value+axis.scale)]

        ssd_range = [float(axis.scale),
                     float(axis.high_value+axis.scale)]

        if ssd_threshold is not None:
            ssd_range[0] = ssd_threshold

        if zlp_threshold is None:
            zlp_threshold  = float(zlp.axis.axis[zlp.channel_switches][-1])

        # EEL is updated separately
        eel = self.deepcopy()

        # progressbar support
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        pbar = progressbar(total = iterations,
                           desc = '1.00e+30',
                           disable=not show_progressbar)

        # initialize iteration control
        io = 0
        chi2 = chi2_target*1e3
        while (io < iterations) and (chi2 > chi2_target):

            # Update ZLP model with appropriate size (expand and crop technique)
            z = eel.model_zero_loss_peak(threshold = zlp_threshold,
                                         model     = zlp,
                                         show_progressbar=False)

            # Calculate the ZLP intensity
            Izlp = z.integrate1D(-1)

            # extract SSD using the Fourier-log deconvolution
            if kwpad is not None:
                # Pad the EELS spectra
                eel = eel.power_law_extrapolation_until(**kwpad)
            ssd = eel.fourier_log_deconvolution(z)
            del eel
            ssd.remove_negative_intensity(inplace=True)
            if kwpad is not None:
                ssd.crop_signal1D(*ssd_range)
            else:
                ssd.crop_signal1D(ssd_range[0], None)
            ssd.data += 1e-6

            # Normalize spectrum and Kramers-Kronig transform
            if kwpad is not None:
                ssd = ssd.power_law_extrapolation_until(**kwpad)
            else:
                ssd.hanning_taper('both')
            ssd, tkka = ssd.normalize_bulk_inelastic_scattering(Izlp, n=n, t=t)
            eps = ssd.kramers_kronig_transform(invert=True)
            if kwpad is not None:
                eps.crop(-1, *ssd_range)
            else:
                eps.crop(-1, ssd_range[0], None)
            del ssd

            # Apply averaging if needed
            eps_corr = eps.deepcopy()
            if average and (eps.axes_manager.navigation_dimension > 0):
                eps_corr.data[:] = eps.data.mean(
                                   eps.axes_manager.navigation_indices_in_array,
                                   keepdims=True)
            else:
                eps_corr.data = eps.data.copy()

            if iterations > 1:
                # Relativistic correction
                #  Calculates relativistic correction from the Kroeger equation
                #  The difference with the relativistic DCS is subtracted
                ssd, vol, elf = eps_corr.get_relativistic_spectrum(
                               zlp=Izlp, t=tkka, output='full', *args, **kwargs)
                del vol

                # Fourier-exp convolution with the current ZLP model...
                # ... uses the expand technique and optional PLaw padding
                # ...for the SSD
                ssd.expand_signal1D(eel_range[0], None, inplace=True)
                if kwpad is not None:
                    ssd = ssd.power_law_extrapolation_until(**kwpad)
                else:
                    ssd.hanning_taper('both')
                ssd = ssd.fourier_exp_convolution(z)
                if kwpad is not None:
                    ssd.crop_signal1D(*eel_range)

                # ... for the ELF
                elf.expand_signal1D(eel_range[0], None, inplace=True)
                if kwpad is not None:
                    elf = elf.power_law_extrapolation_until(**kwpad)
                else:
                    elf.hanning_taper('both')
                elf = elf.fourier_exp_convolution(z)
                if kwpad is not None:
                    elf.crop_signal1D(*eel_range)

                # calculate correction
                scorr = ssd - elf
                del ssd, elf

                # Limit the fractional correction
                if delta is not None:
                    fcorr = np.clip(scorr.data / self.data, -delta, delta)
                    scorr.data = fcorr * self.data

                # Apply correction
                eel = self - scorr
                eel.remove_negative_intensity(inplace=True)

                if io > 0:
                    #chi2 = ((scorr.data-smemory)**2/smemory**2).sum()
                    chi2 = smemory._chi2_score(scorr)
                    chi2str = '{:0.2e}'.format(chi2)
                    if show_progressbar:
                        pbar.set_description(chi2str)
                smemory = scorr.deepcopy()
            io += 1
            pbar.update(1)

        pbar.close()

        output = {}
        tstr = self.metadata.General.title
        tkka.metadata.General.title = 'r-KKA thickness'
        output['thickness'] = tkka
        scorr.metadata.General.title = tstr + 'r-KKA correction'
        output['relativistic correction'] = scorr
        return eps, output

    def _chi2_score(self, sig, p=0.5):
        """
        Calculate the mean chi-2 score removing outliers in a pedestrian way.

        Parameters
        ----------
        sig : signal
         Must be of equal dimensions to self or this won't work.
        p : float
         Remove outliers coefficient: percentage of the data left out above and
         below.

        Returns
        -------
        chi2 : float
         Average of the chi-square test score.
        """
        chi2 = (sig.data - self.data)**2 / self.data**2
        chi2 = np.nan_to_num(chi2)
        a_min, a_max = np.percentile(chi2, p), np.percentile(chi2, 100-p)
        return np.clip(chi2, a_min, a_max).mean()
