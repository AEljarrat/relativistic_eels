import numpy as np
from scipy.interpolate import interp2d
from scipy.integrate import quad
from scipy.special import logsumexp
from multiprocessing import Pool
from cubature import cubature

class interp_quad(object):
    """
    Integrate a 2d array using quadrature.
    The integral is solved over the second dimension (using integrate_y method),
    iteratively over the first dimension. The quad Fortran library is used. If
    Ncpu is provided, and if possible, the iterations run in a parallel pool
    using the multiprocessing library.

    Deprecated! Use the more efficient cubature instead.
    """
    def __init__(self, xaxis, yaxis, dmesh):
        """
        Quadrature integration init. A scipy interpolator is created on object
        instantiation. using the provided dimensions (axes) and mesh.

        Parameters
        ----------
        xaxis : first dimension
        yaxis : second dimension
        dmesh : the data we want to integrate
        """
        self.fmesh = interp2d(xaxis, yaxis, dmesh.T, kind='cubic')
        self.xaxis = xaxis
        self.yaxis = yaxis

    def _ify(self, yvalue):
        """
        Integrator function with quad over the 2nd dimension
        """
        return quad(self.fmesh, self.xaxis.min(), self.xaxis.max(), yvalue)[0]

    def integrate_y(self, Ncpu=1, pchunk=1):
        """
        Integrate iteratively over the 1st dimension.
        """
        if Ncpu > 1:
            # parallel integration
            chunksize = (len(self.yaxis) // Ncpu) // pchunk
            result = np.zeros_like(self.yaxis)
            with Pool(processes=Ncpu) as pool:
                p_iter = pool.imap(self._ify, self.yaxis, chunksize)
                for io, res in enumerate(p_iter):
                    result[io] = res
        else:
            # serial integration
            result = np.zeros_like(self.yaxis)
            for io, iey in enumerate(self.yaxis):
                result[io] = self._ify(iey)

        return result

def interp_cuba(datamesh, xaxis, yaxis):
    '''
    Cubature integration of a 2D data mesh using a cubic interpolator funtion.
    Can be used by the map function to do parallel integration of hyper-images.

    Parameters
    ----------
    datamesh : np.ndarray
     Numpy array containig a 2D data mesh. Dimensions correspond to input arrays
     datamesh.shape = [len(yaxis), len(xaxis)]
    xaxis : np.ndarray
     Numpy vector with correct dimensions.
    yaxis : np.ndarray
     Numpy vector with correct dimensions.

    Returns
    -------
    cuba_integral: np.ndarray
     Numpy vector, same length as yaxis.
    '''
    foomesh = interp2d(xaxis, yaxis, datamesh.T, kind='cubic')
    cuba_integral, err = cubature(foomesh,
                                  ndim=1,
                                  fdim=len(yaxis),
                                  xmin=[xaxis.min()],
                                  xmax=[xaxis.max()],
                                  args=(yaxis,))
    return cuba_integral

def log_sum_exp(logmesh, xaxis, yaxis):
    '''
    Log-sum-exp integration for a logarithmic mesh.

    Parameters
    ----------
    logmesh : np.ndarray
     Numpy array containig a 2D data mesh. Dimensions correspond to input arrays
     datamesh.shape = [len(yaxis), len(xaxis)]
    xaxis : np.ndarray
     Numpy vector with correct dimensions. It follows a geometric progression.
    yaxis : np.ndarray
     Numpy vector with correct dimensions.

    Returns
    -------
    lse_integral: np.ndarray
     Numpy vector, same length as yaxis.
    '''
    lsum, lsig = logsumexp(np.log(logmesh), 0, xaxis[:,None], return_sign=True)
    a_logse = np.log(xaxis)
    da_logse = a_logse[1] - a_logse[0]
    return np.exp(lsum*lsig) * da_logse
