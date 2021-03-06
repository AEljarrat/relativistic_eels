import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import legend as legend_mpl
from hyperspy.signals import Signal1D, BaseSignal
from hyperspy.drawing.signal1d import Signal1DLine

from numbers import Number

import itertools
from distutils.version import LooseVersion

class Signal1DComparator():
    def __init__(self,
                 spectra,
                 color=None,
                 line_style=None,
                 legend=None,
                 legend_picking=True,
                 legend_loc='upper right',
                 navigator='auto',
                 axes_manager=None,
                 **kwargs):
        """Plot several spectra in the same figure, multidimensional version.

        Extra keyword arguments are passed to `hyperspy.signal.plot`.

        Parameters
        ----------
        spectra : iterable object
            Ordered spectra list to plot.
        color : matplotlib color or a list of them or `None`
            Sets the color of the lines of the plots (no action on 'heatmap').
            If a list, if its length is less than the number of spectra to plot,
            the colors will be cycled. If `None`, use default matplotlib color
            cycle.
        line_style: {None, 'scatter', 'line', 'step'}
            Sets the line style of the plots. The only available line style are
            'step','scatter' and 'line'. If a list, if its length is less than
            the number of spectra to plot, line_style will be cycled. If `None`,
            'step' line style will be used for an original hyperspy feeling.
        legend: None or list of str or 'auto'
           If list of string, legend for "cascade" or title for "mosaic" is
           displayed. If 'auto', the title of each spectra (metadata.General.title)
           is used.
        legend_picking: bool
            If true, a spectrum can be toggle on and off by clicking on
            the legended line. Will not work with the 'scatter' line_style.
        legend_loc : str or int
            This parameter controls where the legend is placed on the figure;
            see the pyplot.legend docstring for valid values
        navigator : {"auto", None, "slider", "spectrum", Signal}
            As in `hyperspy.signal.plot`, see docs therein for more info.
        axes_manager : {None, axes_manager}
            If None a proper `axes_manager` is detected and used.
        **kwargs
            remaining keyword arguments are passed to hyperspy.signal.plot()

        Example
        -------
        >>> s = hs.load("some_spectra")
        >>> hs.plot.plot_spectra(s, style='cascade', color='red', padding=0.5)

        To save the plot as a png-file

        >>> hs.plot.plot_spectra(s).figure.savefig("test.png")

        Returns
        -------
        ax: matplotlib axes or list of matplotlib axes
            An array is returned when `style` is "mosaic".

        """

        if isinstance(spectra, Signal1D):
            spectra._check_signal_dimension_equals_one()
            signal_list = [s,]
        elif isinstance(spectra, list):
            signal_list = []
            for s in spectra:
                s._check_signal_dimension_equals_one()
                signal_list += [s,]

        # Take the axes_manager from the first multidimensional signal.
        # If this is not possible, just use the first signal.
        if axes_manager is None:
            multi_spectra = [s for s in signal_list \
                                       if s.axes_manager.navigation_shape != ()]
            if len(multi_spectra) > 0:
                # Check to see if the spectra have the same navigation shapes
                axes_manager = multi_spectra[0].axes_manager
                temp_shape_first = axes_manager.navigation_shape
                for s in multi_spectra:
                    temp_shape = s.axes_manager.navigation_shape
                    if not (temp_shape_first == temp_shape):
                        raise ValueError("The provided spectra do not have the "
                                         "same navigation shape.")
            else:
                axes_manager = signal_list[0].axes_manager

        axdict = axes_manager.as_dictionary()
        axdict = [axdict[keys] for keys in axdict.keys()]
        data = np.zeros(axes_manager.navigation_shape[::-1] +
                         axes_manager.signal_shape)
        self.signal = Signal1D(data=data, axes=axdict)
        self.signal.plot(navigator=navigator)

        line = self.signal._plot.signal_plot.ax_lines[0]
        line.line.set_visible(False)
        line.autoscale = False

        # color from plot_spectra
        if color is not None:
            if isinstance(color, str):
                color = itertools.cycle([color])
            elif hasattr(color, "__iter__"):
                color = itertools.cycle(color)
            else:
                raise ValueError("Color must be None, a valid matplotlib color "
                                 "string or a list of valid matplotlib colors.")
        else:
            if LooseVersion(mpl.__version__) >= "1.5.3":
                color = itertools.cycle(
                    plt.rcParams['axes.prop_cycle'].by_key()["color"])
            else:
                color = itertools.cycle(plt.rcParams['axes.color_cycle'])

        # line_style from plot_spectra
        if line_style is not None:
            if isinstance(line_style, str):
                line_style = itertools.cycle([line_style])
            elif hasattr(line_style, "__iter__"):
                line_style = itertools.cycle(line_style)
            else:
                raise ValueError("line_style must be None, a valid matplotlib"
                                 " line_style string or a list of valid matplotlib"
                                 " line_style.")
        else:
            line_style = ['step'] * len(spectra)


        self.signal_lines = []
        for s, c, t in zip(signal_list, color, line_style):
            self.add_line(s, color=c, type=t)

        self.signal.axes_manager.events.indices_changed.connect(
            self.update_position, [])
        self.update_position()

        if legend is not None:
            if isinstance(legend, str):
                if legend == 'auto':
                    legend = [spec.metadata.General.title for spec in spectra]
                    legend = legend
                else:
                    raise ValueError("legend must be None, 'auto' or a list of"
                                    " string")
        elif hasattr(legend, "__iter__"):
            legend = list(legend)
            legend = itertools.cycle(legend)

        if legend is not None:
            ax = self.signal._plot.signal_plot.ax
            ax.legend(ax.lines[1:], legend, loc=legend_loc)
            if legend_picking is True:
                self.animate_legend()

    def add_line(self, signal, color='b', type='step', fill_with=np.nan):
        axis_plt = self.signal.axes_manager.signal_axes[0]
        axis_sig = signal.axes_manager.signal_axes[0]
        rdata = fill_with * np.ones_like(axis_plt.axis)
        idx = axis_plt.value2index(axis_sig.low_value)
        fdx = axis_plt.value2index(axis_sig.high_value) + 1

        def _data_f(axes_manager):
            rdata[idx:fdx] = signal.__call__(signal.axes_manager)
            return rdata

        def _data_f_multi(axes_manager):
            signal.axes_manager.trait_set(indices=axes_manager.indices)
            return _data_f(axes_manager=axes_manager)

        signal_line = Signal1DLine()
        if signal.axes_manager.navigation_shape == ():
            signal_line.data_function = _data_f
        else:
            signal_line.data_function = _data_f_multi

        signal_line.set_line_properties(color  = color,
                                        type   = type,
                                        scaley = False)
        self.signal._plot.signal_plot.add_line(signal_line)
        signal_line.autoscale = False
        self.signal_lines += [signal_line]
        signal_line.plot()

    def update_position(self, *args, **kwargs):
        axis = self.signal.axes_manager.signal_axes[0]
        ax = self.signal._plot.signal_plot.ax
        ax.relim()
        y1, y2 = np.searchsorted(axis.axis,
                                 ax.get_xbound())
        y2 += 2
        y1, y2 = np.clip((y1, y2), 0, len(axis.axis - 1))

        maxs = []
        mins = []
        for li in ax.lines:
            if li.get_visible():
                di = li.get_ydata()[y1:y2]
                maxs += [np.nanmax(di),]
                mins += [np.nanmin(di),]
        ax.set_ylim(np.min(mins), np.max(maxs))
        self.signal._plot.signal_plot.figure.canvas.draw_idle()

    def animate_legend(self):
        """Animate the legend of a figure.

        A spectrum can be toggle on and off by clicking on the legended line.

        Parameters
        ----------

        figure: 'last' | matplotlib.figure
            If 'last' pick the last figure

        Note
        ----

        Code inspired from legend_picking.py in the matplotlib gallery

        """
        ax = self.signal._plot.signal_plot.ax
        figure = ax.figure
        lines = ax.lines[1:]
        lined = dict()
        leg = ax.get_legend()
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(5)  # 5 pts tolerance
            lined[legline] = origline

        def onpick(event):
            # on the pick event, find the orig line corresponding to the
            # legend proxy line, and toggle the visibility
            legline = event.artist
            origline = lined[legline]
            vis = not origline.get_visible()
            origline.set_visible(vis)
            # Change the alpha on the line in the legend so we can see what lines
            # have been toggled
            if vis:
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.2)
            figure.canvas.draw_idle()

        figure.canvas.mpl_connect('pick_event', onpick)

class SignalMixin(BaseSignal):
    def __init__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], BaseSignal):
            # Pretend it is a hs signal, copy axes and metadata
            sdict = args[0]._to_dictionary()
            self.__class__.__init__(self, **sdict)
        else:
            BaseSignal.__init__(self, *args, **kwargs)

    def _check_adapt_map_input(self, ins, varname=''):
        '''
        Check and adapt an input parameter that will work with the map function
        for this signal. An adapted signal is returned in case it works. If it
        does not work, a meaningful ValueError is returned instead.

        Parameters
        ----------
        ins : {number, ndarray, hs.signals.BaseSignal}
         One of the following:
         -  Single number.
         -  Numpy array with the same number of elements as this signal
         navigation shape.
         -  HyperSpy signal with same navigation shape + null signal dimension,
         or same signal shape as navigation shape + null navigation dimension,
         or same len as this signal, or containing just a single value.

        Returns
        -------
        aus : hs.signals.BaseSignal
         If a single value was provided, this is a signal with navigation shape
         equal to 1. Any other compatible case is transformed into a signal with
         same navigation shape as this signal, and null navigation dimension. In
         case no possible transformation is possible, a ValueError type results.
        '''
        aus = None
        navs = self.axes_manager.navigation_shape

        if isinstance(ins, Number):
            aus = BaseSignal(ins).T
            return aus

        elif isinstance(ins, np.ndarray):
            if ins.shape == navs:
                aus = BaseSignal(ins.T).T
            elif ins.shape == navs[::-1]:
                aus = BaseSignal(ins).T
            elif len(ins) == len(self):
                aus = BaseSignal(ins.reshape(navs[::-1])).T
            else:
                aus = ValueError('Input array not recognized, ')
            return aus

        elif isinstance(ins, BaseSignal):
            if ins.axes_manager.navigation_shape == navs and (
               ins.axes_manager.signal_dimension == 0):
                aus = ins
            elif ins.axes_manager.signal_shape == navs and (
                 ins.axes_manager.navigation_dimension == 0):
                aus = ins.T
            elif ins.data.shape == (1,):
                aus = BaseSignal(ins.data).T
            elif len(ins) == len(self) and (
                 ins.axes_manager.signal_dimension == 0):
                aus = BaseSignal(ins.data.reshape(navs[::-1])).T
            else:
                aus = ValueError('Input signal not recognized, ')
            return aus

        if aus is None:
            return aus
        else:
            aus = ValueError('Input '+type(ins)+' not recognized, ')
            return aus
