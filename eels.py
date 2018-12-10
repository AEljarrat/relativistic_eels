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

    def remove_negative_intensity(self, inplace=False):
        '''
        By definition, electron energy-loss spectral intensity is positive but,
        sometimes, our spectral treatments overlook this important fact.

        Parameters
        ----------
        inplace : bool
         Performs the operation in-place.

        Returns
        -------
        spc : EELSSpectrum
         Negative intensity is set to 0.
        '''
        if inplace:
            spc = self
        else:
            spc = self.deepcopy()
        spc.data[spc.data < 0.] = 0
        return spc

    def remove_intensity(self, left_value=None, right_value=None, inplace=False):
        '''
        Remove all intensity in a range.

        Parameters
        ----------
        threshold : {int, float}
         Integer index or energy-loss units.
        inplace : bool
         Performs the operation in-place.

        Returns
        -------
        spc : EELSSpectrum
         Spectral intensity in this range is set to 0.
        '''
        self._check_signal_dimension_equals_one()
        try:
            signal_range_from_roi = hs.hyperspy.misc.utils.signal_range_from_roi
            left_value, right_value = signal_range_from_roi(left_value)
        except TypeError:
            # It was not a ROI, we carry on
            pass
        eax = self.axes_manager.signal_axes[0]
        i1, i2 = eax._get_index(left_value), eax._get_index(right_value)
        if inplace:
            spc = self
        else:
            spc = self.deepcopy()
        spc.data[(slice(None),)*eax.index_in_array+(slice(i1, i2),Ellipsis)]=0.
        return spc

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

    def model_zero_loss_peak(self, signal_range, model=None, replace_data=True):
        """
        Flexible tool to model and fit the ZLP.

        Parameters
        ----------
        signal_range : {list, tuple}
         With two values, for the lower and upper limits of the range in which
         the zero-loss peak model will be fitted. The first value may be None,
         then it is set to the first value of the signal axis.
        model : hyperspy model
         To be used to fit the zero-loss peak. If None is provided, a Voigt peak
         is used by default. Note that the model.signal parameter is substituted
         by the input signal.
        replace_data : bool
         Replace the data below the second value of the signal_range with the
         experimental data.

        Returns
        -------
        zlp : ModifiedEELS
         A zero-loss peak model with the same dimensions as the input signal. It
         can be used by subtraction or deconvolution.
        """

        self._check_signal_dimension_equals_one()
        if not isinstance(signal_range, tuple):
            raise AttributeError('signal_range not recognized:'
                                 'must be a tuple!')

        if len(signal_range) != 2:
            raise AttributeError('signal_range not recognized,'
                                 'must be len = 2')

        if model is None:
            # default to Voigt peak
            model = self.create_model(auto_background = False,
                                      auto_add_edges  = False)
            model.append(hs.model.components1D.Voigt())

        if model.signal is not self:
            model.signal = self

        model.set_signal_range(*signal_range)
        model.multifit(show_progressbar=False)

        # Create zlp model (use only the tail from the fit)
        model.reset_signal_range()
        zlp = model.as_signal(show_progressbar=False)
        model.set_signal_range(*signal_range)

        if replace_data:
            # trust the data below the fitting limit
            idx = self.axes_manager.signal_axes[0].value2index(signal_range[1])
            zlp.data[..., :idx] = self.data[..., :idx]

        return zlp

    def power_law_extrapolation_until(self, window_size=20, total_size=1024,
                                      hanning=True, *args, **kwargs):
        '''
        Usefullity mathod. Extends the spectrum using `power_law_extrapolation`
        until the resulting spectrum has the given total number of pixels.

        Parameters
        ----------
        window_size : {int, float}
         Window size. Alternatively, this parameter can be a float specifying
         the size in energy-loss units.
        total_size : {int, float}
         Total number of pixels. Alternatively, this parameter can be a float
         specifying the size in energy-loss units.

        Returns
        -------
        spc : EELSSpectrum
         The extended spectrum.

        '''
        eax = self.axes_manager.signal_axes[0]

        if type(total_size) is float:
            offset = self.axes_manager[-1].offset - self.axes_manager[-1].scale
            total_size = int(round((total_size-offset)/eax.scale))

        if type(window_size) is float:
            window_size = int(round(window_size/eax.scale))

        extrapolation_size = total_size-eax.size
        if extrapolation_size < 0:
            raise AttributeError("total_size is less than spectral axis size")

        spc = self.power_law_extrapolation(window_size,
                                           extrapolation_size,
                                           *args,
                                           **kwargs)

        if hanning:
            spc.hanning_taper('both')

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

        Returns
        -------
        elf: ModifiedEELS
            The imaginary part of the inverse dielectric function, also known as
            energy-loss function.
        t : Signal
            The thickness, estimated from the refractive index in case this was
            provided. In case the thickness was already provided, we have made
            sure to adapt the navigation dimensions of the input signals.
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
                                        fsmooth=None,
                                        iterations=20,
                                        chi2_target=1e-4,
                                        zlp_range=None,
                                        ssd_range=(0.5, None),
                                        average=False,
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
        zlp: EELSModel
            ZLP model containing a Voigt component to fit the zlp.
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

        # The spectrum contains the zero-loss peak
        Izlp = self.estimate_elastic_scattering_intensity(zlp_range[1])

        # update the eels separately
        eels = self.deepcopy()

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

            # Update ZLP model
            z = eels.model_zero_loss_peak(signal_range = zlp_range,
                                          model        = zlp,
                                          replace_data = False)

            # extract SSD using the Fourier-log deconvolution
            ssd = eels.fourier_log_deconvolution(z)
            ssd.remove_negative_intensity(inplace=True)
            ssd.crop_signal1D(*ssd_range)
            ssd.data += 1e-6

            if ssd_range[1] is None:
                ssd.hanning_taper()
            else:
                Efin = float(self.axes_manager[-1].high_value)
                ssd = ssd.power_law_extrapolation_until(5., Efin,
                                                 fix_neg_r=True, add_noise=True)

            # Normalize spectrum and Kramers-Kronig transform
            ssd, tkka = ssd.normalize_bulk_inelastic_scattering(Izlp,n=n,t=t)
            eps = ssd.kramers_kronig_transform(invert=True)
            del eels, ssd # release memory

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

                # Extend data range to fit the whole EELS set, including ZLP
                Eeels = self.axes_manager[-1].offset
                Essd  = ssd.axes_manager[-1].offset
                dshift = (Essd - Eeels) * np.ones(
                                    ssd.axes_manager._navigation_shape_in_array)

                kwerps = {'shift_array' : dshift,
                          'crop' : False,
                          'expand': True,
                          'fill_value': 0.,
                          'show_progressbar' : False}

                ssd.shift1D(**kwerps)
                elf.shift1D(**kwerps)
                ssd.axes_manager[-1].offset = Eeels
                elf.axes_manager[-1].offset = Eeels

                # Fourier-exp convolution
                from model import fourier_exp_convolution
                ssd = fourier_exp_convolution(ssd, z)
                elf = fourier_exp_convolution(elf, z)

                # calculate correction
                scorr = ssd - elf
                del ssd, elf

                # Limit the fractional correction
                if delta is not None:
                    fcorr = np.clip(scorr.data / self.data, -delta, delta)
                    scorr.data = fcorr * self.data

                # smooth, already smoothed?
                #if fsmooth is not None:
                #    scorr.gaussian_filter(fsmooth)

                # Apply correction
                # debug macro!
                #import ipdb; import matplotlib.pyplot as plt
                #def plots(some_signal):
                #    some_signal.plot()
                #    plt.pause(1)
                #ipdb.set_trace()
                eels = self - scorr
                eels.remove_negative_intensity(inplace=True)

                if io > 0:
                    #chi2 = ((scorr.data-smemory)**2/smemory**2).sum()
                    chi2 = smemory._chi2_score(scorr)
                    chi2str = '{:0.2e}'.format(chi2)
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
