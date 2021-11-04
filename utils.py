import numpy as np
import matplotlib.pyplot as plt
import time
import math
import pyfftw
from scipy import integrate
from scipy.interpolate import interp2d


class Gnlse:
    # ivp solver params, change tolerances if you dont get satisfied with
    rtoll = 1e-4
    atoll = 1e-4
    methodd = 'RK45'

    def __init__(self, fiber_length=1, peak_power=3e3, wavelength=1550, wlRange=None, betas=np.array([
        -1.28e-3, -7.66e-6, 0, 0, 0, 0,
        0, 0, 0
    ]), nonLinearity=0.0108, loss=0, z_saves=100, N=2 ** 12, time_window=12.5,
                 duration=0.200, self_steepening=True):

        if wlRange is None:
            wlRange = [1545, 1555.5]
        self.c = 299792.458  # Speed of light in vacuum [nm/ps]
        self.wlRange = wlRange
        # simulation parameters
        self.fiber_length = fiber_length  # meter
        self.z_saves = z_saves  # number of saves in space

        # Number of time points
        self.N = N

        self.time_window = time_window  # ps
        self.wavelength = wavelength  # nm

        # Input impulse parameters
        self.peak_power = peak_power  # W
        self.duration = duration  # ps
        self.self_steepening = self_steepening
        self.nonlinearity = nonLinearity  # 1/W/m for SMF28 at 1550nm
        self.loss = loss

        self.betas = betas  # ps^2/m, ps**3/m

        # Time domain grid
        self.t = np.linspace(-self.time_window / 2, self.time_window / 2, N)

        # Relative angular frequency grid
        self.V = 2 * np.pi * np.arange(-self.N / 2, self.N / 2) / (self.N * (self.t[1] - self.t[0]))
        # Central angular frequency [10^12 rad]
        self.w_0 = (2.0 * np.pi * self.c) / wavelength
        self.Omega = self.V + self.w_0
        self.wl = 2 * np.pi * self.c / (self.Omega)

        self.Riis = np.where(
            (self.wl > wlRange[0]) & (self.wl < wlRange[1]))  # indices of interest for spectral modulation

        # Absolute angular frequency grid
        if self.self_steepening and np.abs(self.w_0) > np.finfo(float).eps:
            self.W = self.V + self.w_0
        else:
            self.W = np.full(self.V.shape, self.w_0)

        self.W = np.fft.fftshift(self.W)

        # Nonlinearity

        self.gamma = self.nonlinearity / self.w_0
        self.scale = 1

        self.D = self.Dispersion()

        self.dt = self.t[1] - self.t[0]
        self.D = np.fft.fftshift(self.D)
        self.x = pyfftw.empty_aligned(N, dtype="complex128")
        self.X = pyfftw.empty_aligned(N, dtype="complex128")
        self.plan_forward = pyfftw.FFTW(self.x, self.X)
        self.plan_inverse = pyfftw.FFTW(self.X, self.x, direction="FFTW_BACKWARD")

        self.Z = np.linspace(0, self.fiber_length, self.z_saves)

        self.fr, self.RT = self.raman_blowwood(self.t)

        self.RW = self.N * np.fft.ifft(np.fft.fftshift(np.transpose(self.RT)))

    def Dispersion(self):
        # Damping
        alpha = np.log(10 ** (self.loss / 10))
        # Taylor series for subsequent derivatives
        # of constant propagation
        B = sum(beta / np.math.factorial(i + 2) * self.V ** (i + 2)
                for i, beta in enumerate(self.betas))
        L = 1j * B - alpha / 2
        return L

    def raman_blowwood(self, T):
        """Raman scattering function for silica optical fibers, based on K. J. Blow
        and D. Wood model.
        Parameters
        ----------
        T : float
           Time vector.
        Returns
        -------
        fr : float
           Share of Raman response.
        RT : ndarray
           Vector representing Raman response.
        """

        # Raman response [arbitrary units]
        fr = 0.18
        # Adjustable parameters used to fit the actual Raman gain spectrum [ps]
        tau1 = 0.0122
        tau2 = 0.032
        # Raman response function
        ha = (tau1 ** 2 + tau2 ** 2) / tau1 / (tau2 ** 2) * np.exp(-T / tau2) * np.sin(
            T / tau1)
        RT = ha

        RT[T < 0] = 0

        return fr, RT

    def rhs(self, z, AW):

        """
        The right hand side of the differential equation to integrate.
        """

        self.x[:] = AW * np.exp(self.D * z)
        At = self.plan_forward().copy()
        IT = np.abs(At) ** 2

        if self.RW is not None:
            self.X[:] = IT
            self.plan_inverse()
            self.x[:] *= self.RW
            self.plan_forward()
            RS = self.dt * self.fr * self.X
            self.X[:] = At * ((1 - self.fr) * IT + RS)
            M = self.plan_inverse()
        else:
            self.X[:] = At * IT
            M = self.plan_inverse()

        rv = 1j * self.gamma * self.W * M * np.exp(-self.D * z)

        return rv

    def run(self, A):
        # ivp solver params, change tolerances if you dont get satisfied with
        rtoll = 1e-4
        atoll = 1e-4
        methodd = 'RK45'

        solution = integrate.solve_ivp(self.rhs, t_span=(0, self.fiber_length), y0=np.fft.ifft(A), method=methodd,
                                       t_eval=self.Z, rtol=rtoll, atol=atoll)
        AW = solution.y.T

        # Transform the results into the time domain
        At = np.zeros(AW.shape, dtype=AW.dtype)
        for i in range(len(AW[:, 0])):
            AW[i, :] *= np.exp(np.transpose(self.D) * self.Z[i]) / self.scale
            At[i, :] = np.fft.fft(AW[i, :])
            AW[i, :] = np.fft.fftshift(AW[i, :]) * self.N * self.dt

        return AW, At

    def gaussianPULSE(self):

        # Input impulse
        m = 4 * np.log(2)
        impulse_model = np.sqrt(self.peak_power) * np.exp(-m * .5 * self.t ** 2 / self.duration ** 2)

        return impulse_model

    def plot_wavelength_vs_distance(self, AW, WL_range=[400, 1350], ax=None,
                                    norm=None):
        """Plotting results in frequency (wavelength) domain.
        Parameters
        ----------
        solver : Solution
            Model outputs in the form of a ``Solution`` object.
        WL_range : list, (2, )
            Wavelength range. Set [400, 1350] as default.
        ax : :class:`~matplotlib.axes.Axes`
            :class:`~matplotlib.axes.Axes` instance for plotting
        norm : float
            Normalization factor for output spectrum. As default maximum of
            square absolute of ``solver.AW`` variable is taken.
        Returns
        -------
        ax : :class:`~matplotlib.axes.Axes`
          Used :class:`~matplotlib.axes.Axes` instance.
        """
        W = self.Omega
        if ax is None:
            ax = plt.gca()

        if norm is None:
            norm = np.max(np.abs(AW) ** 2)

        lIW = np.fliplr(
            10 * np.log10(np.abs(AW) ** 2 / norm,
                          where=(np.abs(AW) ** 2 > 0)))
        WL = 2 * np.pi * self.c / W  # wavelength grid
        WL_asc = np.flip(WL, )  # ascending order for interpolation
        iis = np.logical_and(WL_asc > WL_range[0],
                             WL_asc < WL_range[1])  # indices of interest

        WL_asc = WL_asc[iis]
        lIW = lIW[:, iis]

        interpolator = interp2d(WL_asc, self.Z, lIW)
        newWL = np.linspace(np.min(WL_asc), np.max(WL_asc), lIW.shape[1])
        toshow = interpolator(newWL, self.Z)

        ax.imshow(toshow, origin='lower', aspect='auto', cmap="magma",
                  extent=[np.min(WL_asc), np.max(WL_asc), 0, np.max(self.Z)],
                  vmin=-40)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Distance [m]")
        return ax

    def plot_delay_vs_distance(self,At, time_range=None, ax=None, norm=None):
        """Plotting results in time domain.
        Parameters
        ----------
        solver : Solution
            Model outputs in the form of a ``Solution`` object.
        time_range : list, (2, )
            Time range. Set [min(``solver.t``), max(``solver.t``)] as default.
        ax : :class:`~matplotlib.axes.Axes`
            :class:`~matplotlib.axes.Axes` instance for plotting.
        norm : float
            Normalization factor for output spectrum. As default maximum of
            square absolute of ``solver.At`` variable is taken.
        Returns
        -------
        ax : :class:`~matplotlib.axes.Axes`
          Used :class:`~matplotlib.axes.Axes` instance.
        """
        if ax is None:
            ax = plt.gca()

        if time_range is None:
            time_range = [np.min(self.t), np.max(self.t)]

        if norm is None:
            norm = np.max(np.abs(At) ** 2)

        lIT = 10 * np.log10(np.abs(At) ** 2 / norm,
                            where=(np.abs(At) ** 2 > 0))

        ax.pcolormesh(self.t, self.Z, lIT, shading="auto", vmin=-40,
                      cmap="magma")
        ax.set_xlim(time_range)
        ax.set_xlabel("Delay [ps]")
        ax.set_ylabel("Distance [m]")

        return ax

    def checkWLrange(self):
        '''
        Checks wavelength boundaries for given time window and sampling points
        '''

        lambda_min = 1 / (1 / (2 * self.c * self.dt) + 1 / self.wavelength)
        print('lambda min :', lambda_min)
        lambda_max = 1 / (1 / self.wavelength - 1 / (2 * self.c * self.dt))
        print('lambda max :', lambda_max)

        if lambda_max > self.wavelength > lambda_min:
            print('Simulation parameters are ok')
        else:
            print('!!!UPDATE Simulation parameters !!!')

    def amplitudeEncode(self, mask, A):
        InpW = np.fft.fftshift(np.fft.fft(A))
        InpW[self.Riis] = InpW[self.Riis] * mask
        return np.fft.ifft(np.fft.fftshift(InpW))

    def maskdetails(self):
        print('Applied mask WL Range', self.wlRange)
        print('Number of data point on the range', len(self.Riis[0]))
        print('Occupied BW per data element [nm] :', (self.wlRange[1] - self.wlRange[0]) / len(self.Riis[0]))


class NonlinearTransformer(BaseEstimator, TransformerMixin):
    '''
    Transforms input data by propagating it through nonlinear medium
    '''
    # Class Constructor
    def __init__(self):

        # Return self nothing else to do here

    def fit(self, X, y=None):
        return self

        # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        return X[self._feature_names]
