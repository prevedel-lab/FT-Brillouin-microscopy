# -*- coding: utf-8 -*-

import io
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

import scipy
from scipy.optimize import curve_fit

import plotly.io as pio
pio.renderers.default = 'svg'
# pio.renderers.default='browser'


def plotly_to_svg(fig, file, **kwargs):
    """
    Save the plotly fig to svg file and remove the vector-effect: non-scaling-stroke
    (code taken from https://github.com/plotly/plotly.py/issues/1539)

    Parameters
    ----------
    fig : Plotly fig
        The figure to be saved.
    file : String
        The filename of the output SVG file.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    buffer = io.BytesIO()
    fig.write_image(buffer, format="svg", **kwargs)
    with open(file, "bw") as f:
        f.write(
            buffer.getvalue().replace(
                b"vector-effect: non-scaling-stroke", b"vector-effect: none"
            )
        )


if __name__ == "__main__":

    rndGenerator = np.random.default_rng()

    # %% simulation parameters and function definition

    wavelength = 780.032*1e-3  # um

    # whether to use a Lorentzian lineshape or the broad one due to finite NA
    use_broad_lineshape = True
    # the standard deviation of the gaussian which represents the angular distribution due to the NA
    NA_sigma = 0.18

    # number of small steps
    n = 5
    # small step
    s = 0.0975  # um
    # number of large steps
    N = 20
    # large step
    S = 5e3  # um

    # accuracy of the stage on large steps
    # N.B. this is different from the accuracy/precision on small steps
    # and it doesn't affect the results. It is introduced just to avoid
    # periodicity in the phase that is not experimentally the case
    pos_err = 0.1  # um

    # The values for the shift and width are the averages of the experimental values
    shift_H2O = 3.49  # GHz
    width_H2O = 0.15 if use_broad_lineshape else 1.6  # GHz

    # number of repetitions for the Montecarlo simulation
    n_rep = 100

    def fit_cos(pos, I, get_error_on_parameters=False):
        """
        Fit A*cos(2k*pos+phi)+b to I, assuming k=2*pi/wavelength

        Parameters
        ----------
        pos : ndarray
            Positions (um)
        I : ndarray
            Intensity (a.u.)
        get_error_on_parameters : Bool
            Determines if the function should estimate the errors on the parameters

        Returns
        -------
        popt : ndarray
            List of fit parameters [A, 2*k, phi, b]
        perr : ndarray
            List of errors on the fit parameters (if get_error_on_parameters=True, otherwise the array is filled with 1s)

        """
        k = 2*np.pi/wavelength  # um^-1

        # using the formulas in https://en.wikipedia.org/wiki/Least_squares#Linear_least_squares
        # Phi1=Cos(x), Phi2=Sin(x), Phi3=1
        x1 = np.cos(2*k*pos)
        x2 = np.sin(2*k*pos)
        x3 = np.ones(len(pos))

        Xt = np.stack((x1, x2, x3))
        X = Xt.T

        Xt_X_inv = np.linalg.inv(Xt @ X)
        LSM = Xt_X_inv @ Xt
        B = LSM @ I

        popt = np.empty(4)
        popt[0] = np.sqrt(B[0]**2+B[1]**2)  # A
        popt[1] = 2*k
        if B[0] == 0:
            phi = np.pi
        else:
            phi = np.arctan(-B[1]/B[0])
        if B[0] < 0:
            phi += np.pi
        if phi > np.pi:
            phi -= 2*np.pi
        popt[2] = phi
        popt[3] = B[2]  # b

        perr = np.ones_like(popt)
        if get_error_on_parameters:
            rss = np.sum((I - X @ B)**2)
            sigma_squared = rss/(len(I)-Xt.shape[0])
            cov = Xt_X_inv
            var = sigma_squared*np.diag(cov)

            perr[0] = np.sqrt(B[0]**2*var[0] + B[1]**2*var[1])/popt[0]
            perr[1] = 0
            den = B[0]**2 + B[1]**2
            perr[2] = np.sqrt(B[1]**2*var[0] + B[0]**2*var[1])/den
            perr[3] = np.sqrt(var[2])

        return popt, perr

    def _single_peak_fft_Lor_jacob(tao_ns, shift_GHz, width_GHz, amplitude, b):
        f = _single_peak_fft_Lor(tao_ns, shift_GHz, width_GHz, amplitude, 0)
        ds = -2*np.pi*tao_ns*amplitude * \
            np.exp(-np.pi*(tao_ns)*width_GHz) * \
            np.sin(2*np.pi*shift_GHz*(tao_ns))
        dw = -2*(tao_ns)*f
        dA = f/amplitude
        db = np.ones(len(tao_ns))
        return np.stack((ds, dw, dA, db)).T

    def _single_peak_fft_Lor(tao_ns, shift_GHz, width_GHz, amplitude, b):
        return amplitude*np.exp(-np.pi*(tao_ns)*width_GHz)*np.cos(2*np.pi*shift_GHz*(tao_ns))+b

    def _single_peak_fft_broad(tao_ns, shift_GHz, width_GHz, amplitude, b, sigma=NA_sigma):
        """
        The analytical solution of the integral over all the possible angles,
        in the approximation of small angles around pi/2 and gaussian distribution.
        sigma is the width of the gaussian distribution of half the scattering angles around pi/4
        """

        D = 2*np.pi*width_GHz
        s = 2*np.pi*shift_GHz

        A = np.exp(sigma**2*tao_ns**2/2*(D**2-s**2) - D/2*tao_ns)
        return amplitude*A*np.cos(s*tao_ns-sigma**2*tao_ns**2/2*D*s)+b

    _single_peak_fft = _single_peak_fft_Lor
    _single_peak_fft_jacob = _single_peak_fft_Lor_jacob
    if use_broad_lineshape:
        _single_peak_fft = _single_peak_fft_broad
        _single_peak_fft_jacob = None

    def _calculate_FWHM_broadened_lineshape(shift, width):
        """
        Calculate the FWHM of the broadened peak, under the approximation of a Voigt profile,
        based on the parameters fitted by '_single_peak_fft_broad'
        """
        # formula for FWHM taken from https://en.wikipedia.org/wiki/Voigt_profile
        fg = 2.355*NA_sigma*np.sqrt(shift**2-width**2)
        FWHM = 0.5346*width+np.sqrt(0.2166*width**2+fg**2)
        return FWHM

    def fit_single_peak_spectrum(S, A):
        tao_ns = 2e3*S*np.arange(len(A), dtype=np.float64)/scipy.constants.c

        b0 = np.mean(A)
        A0 = np.max(A)-np.min(A)
        """
        p_low = [0, 0, 0, 0]
        p_up = [+np.inf, +np.inf, +np.inf, +np.inf]
        bounds = (p_low, p_up)
        """
        try:
            popt, _ = curve_fit(_single_peak_fft, tao_ns, A, p0=[
                                shift_H2O, width_H2O, A0, b0], jac=_single_peak_fft_jacob)
        except Exception:
            popt = np.full(4, np.nan)
        return popt[0], popt[1]

    # return the signal for a single Lorentzian peak of the given shift and width, where the Brillouin part is normalised to 1

    def signal_single_peak(pos_um, shift_GHz, width_GHz, Rayleigh_intensity=0, ASE=0):
        # time delay between the two arms of the Michelson (in ns)
        tao = np.abs(2e3*pos_um/scipy.constants.c)
        # amplitude for the envelope; the maximum value is Rayleigh_intensity+1
        A = Rayleigh_intensity + \
            _single_peak_fft(tao, shift_GHz, width_GHz, 1, 0)
        return 0.5*(ASE+Rayleigh_intensity+1)+0.5*A*np.cos(4*np.pi*pos_um/wavelength)

    # determine the reference phase
    def get_ref_phase(pos_um):
        return 2*np.pi*((2*pos_um[0]/wavelength) % 1)

    def get_amplitude_at_single_position(func, par1, par2, large_displ_um, n=n, s=s):
        """
        Calculate the shift and width data by using func to generate the interferogram

        Parameters
        ----------
        func : function
            The function used to generate the inteferogram data.
            It must have the signature func(pos, par1, par2) where
                pos is the position in um 
                par1 and par2 are two parameters passed to func
            Returns pos, I
        par1 : double
            See description of "func"
        par2 : double
            See description of "func"
        large_displac_um : double
            The optical path difference between the two paths of the Michelson
        n : double
            Number of small displacements
        s : double
            Step for the small displacements in um

        Returns
        -------
        A: double
            The amplitude that results from the fit of the simulated data

        """
        # add an error on the large steps (see the definition of pos_err for explanation)
        pos = rndGenerator.normal(large_displ_um, pos_err) + s*np.arange(n)

        pos, I = func(pos, par1, par2)

        pos_rel = pos - pos[0]
        popt, perr = fit_cos(pos_rel, I)

        A = popt[0]

        ref_phase = get_ref_phase(pos)
        ph_diff = np.angle(np.exp(1j*popt[2])*np.exp(-1j*ref_phase))
        if np.abs(ph_diff) > np.pi/2:
            A = - A
        return A

    def generate_simulated_data(func, par1_array, par2_array):
        """
        Calculate the shift and width data by using func to generate the interferogram

        Parameters
        ----------
        func : function
            The function used to generate the inteferogram data.
            It must have the signature func(pos, par1, par2) where
                pos is the position in um 
                par1 and par2 are the elements of par1_array and par2_array
            Returns pos, I
        par1 : ndarray
            See description of "func"
        par2 : ndarray
            See description of "func"

        Returns
        -------
        shift_std: ndarray
            The shape is (len(par1_array), len(par2_array)) and contains the
            standard deviation of the shift as measured from the generated interferogram
        width_std: ndarray
            Same as "shift_std" but for the width

        """
        shift_std = np.empty([len(par1_array), len(par2_array)])
        width_std = np.empty_like(shift_std)

        shift = np.empty(n_rep)
        width = np.empty_like(shift)
        A = np.empty(N)
        for p1 in tqdm(range(len(par1_array))):
            for p2 in range(len(par2_array)):
                for i in range(len(shift)):
                    for ld in range(N):
                        A[ld] = get_amplitude_at_single_position(
                            func, par1_array[p1], par2_array[p2], S*ld, n, s)

                    shift[i], width[i] = fit_single_peak_spectrum(S, A)
                shift_std[p1, p2] = np.std(shift)
                width_std[p1, p2] = np.std(
                    _calculate_FWHM_broadened_lineshape(shift, width))

        return shift_std, width_std

    def plot_simulated_data(par1, par2, data, plot_title, xaxis_title, yaxis_title, legend_title, legend_format='{:.2f}', add_linear_fit=False):
        fig = go.Figure()

        colors = px.colors.qualitative.D3

        for j in range(data.shape[1]):
            fig.add_trace(go.Scatter(x=par1, y=data[:, j], name=legend_format.format(
                par2[j]), mode='lines+markers', marker_color=colors[j], marker_size=6, line_color=colors[j], line_width=1.6))

        if add_linear_fit:
            x = par1
            y = data[:, 0]
            l = len(x)
            x = x[l//2:]
            y = y[l//2:]
            coef = np.polynomial.polynomial.Polynomial.fit(
                np.log10(x), np.log10(y), 1).convert().coef
            y_fit = np.power(10.0, coef[0])*np.power(par1, coef[1])
            fig.add_trace(go.Scatter(x=par1, y=y_fit, mode='lines',
                          showlegend=False, line=dict(color='red', width=1.3)))
            i_label = int(l*0.75)
            fig.add_annotation(x=np.log10(par1[i_label]), y=0.4+np.log10(y_fit[i_label]), text="<b>Slope={:.2}</b>".format(
                coef[1]), showarrow=False, font=dict(color='red', size=18, family='Arial'))

        fig.update_layout(
            template='plotly_white',
            legend_title_text=legend_title,
            title=plot_title, title_x=0.5,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            font=dict(
                family="Arial",
                size=22,
                color="Black"
            )
        )
        fig.update_xaxes(type="log", minor=dict(ticks="inside", ticklen=6, showgrid=True),
                         gridcolor='LightSlateGray', gridwidth=0.25,
                         showline=True, linecolor='black', linewidth=2, mirror=True)
        fig.update_yaxes(type="log", minor=dict(ticks="inside", ticklen=6, showgrid=True),
                         gridcolor='LightSlateGray', gridwidth=0.25,
                         showline=True, linecolor='black', linewidth=2, mirror=True)

        # fig.add_vline(x=1e5, line_width=3, line_dash="dash", line_color="green")

        return fig

    def add_experimental_points(fig, x, y, err_x=None, err_y=None):
        fig.add_trace(go.Scatter(
            x=x, y=y,
            name='Experimental',
            error_x=dict(
                type='data',  # value of error bar given in data coordinates
                array=err_x,
                visible=True),
            error_y=dict(
                type='data',  # value of error bar given in data coordinates
                array=err_y,
                visible=True),
            mode='markers', marker_symbol='star-open-dot',
            marker_color='MidnightBlue', marker=dict(size=16,
                                                     line=dict(width=1.2))
        ))
        return fig

    # %% precision at different photoelectrons and camera noise
    N_electrons = np.logspace(np.log10(5), np.log10(1e4), 40)
    # np.logspace(np.log10(0.1), np.log10(10), 4)
    camera_noise = np.array([0.6, 1.4, 5])

    def func(pos, N_electrons, camera_noise):
        I = signal_single_peak(pos, shift_H2O, width_H2O)
        I = rndGenerator.poisson(
            N_electrons*I).astype(np.float64)  # shot noise
        I = rndGenerator.normal(I+5*camera_noise, camera_noise)
        return pos, I

    shift_std, width_std = generate_simulated_data(
        func, N_electrons, camera_noise)

    title = "Shift precision vs. Number of photoelectrons"
    fig = plot_simulated_data(N_electrons, camera_noise, 1e3*shift_std, add_linear_fit=True,
                              plot_title=title, xaxis_title="Number of photoelectrons", yaxis_title="Precision (MHz)",
                              legend_title="Camera noise", legend_format='{:.1f}e-')

    # the number of photons is calculated from the average intensity of the stack
    # minus the offset (100) times the conversion factor (0.49e-/count)
    # the factor 2 is due to the definition used in the simulation of number of
    # photons as the maximum (not the average)
    N_experimental = 2 * 0.49*(np.array([149, 207, 334, 532, 948])-100)
    precision_shift_experimental = np.array([93, 59, 42, 30, 22])
    fig = add_experimental_points(
        fig, N_experimental, precision_shift_experimental)

    fig.show()
    plotly_to_svg(fig, 'shift precision - camera noise.svg')

    title = "Width precision vs. Number of photoelectrons"
    fig = plot_simulated_data(N_electrons, camera_noise, 1e3*width_std, add_linear_fit=True,
                              plot_title=title, xaxis_title="Number of photoelectrons", yaxis_title="Precision (MHz)",
                              legend_title="Camera noise", legend_format='{:.1f}e-')

    precision_width_experimental = np.array([143, 97, 72, 54, 42])
    fig = add_experimental_points(
        fig, N_experimental, precision_width_experimental)

    fig.show()
    plotly_to_svg(fig, 'width precision - camera noise.svg')

    # %% precision at different photoelectrons and stage precision
    N_electrons = np.logspace(np.log10(5), np.log10(1e4), 40)
    stage_precision = np.logspace(np.log10(0.010), np.log10(0.05), 3)

    def func(pos, N_electrons, stage_precision):
        pos = rndGenerator.normal(pos, stage_precision)
        I = signal_single_peak(pos, shift_H2O, width_H2O)
        I = rndGenerator.poisson(
            N_electrons*I).astype(np.float64)  # shot noise
        return pos, I

    shift_std, width_std = generate_simulated_data(
        func, N_electrons, stage_precision)

    title = "Shift precision vs. Number of photoelectrons"
    fig = plot_simulated_data(N_electrons, (1e3*stage_precision).astype(int), 1e3*shift_std, add_linear_fit=False,
                              plot_title=title, xaxis_title="Number of photoelectrons", yaxis_title="Precision (MHz)",
                              legend_title="Stage<br>precision", legend_format='{}nm')
    fig.show()
    plotly_to_svg(fig, 'shift precision - stage precision.svg')

    # %% precision at different photoelectrons and Rayleigh intensity

    N_electrons = np.logspace(np.log10(30), np.log10(1e4), 5)
    Rayleigh_intensity = np.logspace(np.log10(0.1), np.log10(1e3), 40)

    def func(pos, N_electrons, Rayleigh_intensity):
        I = signal_single_peak(pos, shift_H2O, width_H2O,
                               Rayleigh_intensity=Rayleigh_intensity)
        I = rndGenerator.poisson(
            N_electrons*I).astype(np.float64)  # shot noise
        return pos, I

    shift_std, width_std = generate_simulated_data(
        func, N_electrons, Rayleigh_intensity)

    title = "Shift precision vs. Rayleigh/ASE intensity"
    fig = plot_simulated_data(Rayleigh_intensity, N_electrons, 1e3*shift_std.T, add_linear_fit=False,
                              plot_title=title, xaxis_title="Rayleigh/ASE intensity", yaxis_title="Precision (MHz)",
                              legend_title="Number of<br>Brillouin<br>photoelectrons", legend_format='{:.1f}e-')
    fig.show()
    plotly_to_svg(fig, 'shift precision - ASE.svg')

    # %% precision at different photoelectrons and ASE total intensity

    # The ASE plot is a mirror of the Rayleigh plot, so we can merge the two of them

    # N_electrons = np.logspace(np.log10(30), np.log10(1e4), 5)
    # ASE_intensity = np.logspace(np.log10(0.1), np.log10(1e3), 50)

    # def func(pos, N_electrons, ASE_intensity):
    #     I = signal_single_peak(pos, shift_H2O, width_H2O, ASE=ASE_intensity)
    #     I = rndGenerator.poisson(N_electrons*I).astype(np.float64) #shot noise
    #     return pos, I

    # shift_std, width_std = generate_simulated_data(func, N_electrons, ASE_intensity)

    # title = "Shift precision vs. ASE intensity"
    # fig = plot_simulated_data(ASE_intensity, N_electrons, 1e3*shift_std.T, add_linear_fit=False,
    #                     plot_title=title, xaxis_title="ASE intensity", yaxis_title="Precision (MHz)",
    #                     legend_title="Number of Brillouin<br>photoelectrons", legend_format='{:.1f}e-')
    # fig.show()

    # %% precision at different photoelectrons and intensity noise of the laser
    N_electrons = np.logspace(np.log10(5), 4, 40)
    intensity_noise = np.logspace(np.log10(0.002), np.log10(0.2), 5)

    def func(pos, N_electrons, intensity_noise):
        I = signal_single_peak(pos, shift_H2O, width_H2O)
        I = rndGenerator.normal(I, I*intensity_noise)
        I = rndGenerator.poisson(
            N_electrons*I).astype(np.float64)  # shot noise
        return pos, I

    shift_std, width_std = generate_simulated_data(
        func, N_electrons, intensity_noise)

    title = "Shift precision vs. Number of photoelectrons"
    fig = plot_simulated_data(N_electrons, 100*intensity_noise, 1e3*shift_std, add_linear_fit=False,
                              plot_title=title, xaxis_title="Number of photoelectrons", yaxis_title="Precision (MHz)",
                              legend_title="Intensity<br>noise", legend_format='{:.1f}%')
    fig.show()
    plotly_to_svg(fig, 'shift precision - intensity noise.svg')

    # %% precision at different photoelectrons and number of sampling points
    N_electrons = np.logspace(np.log10(5), 4, 5)
    N_sampling_points = np.round(np.logspace(
        np.log10(10), np.log10(1000), 30)).astype(np.int16)

    def func(pos, N_electrons, _):
        I = signal_single_peak(pos, shift_H2O, width_H2O)
        I = rndGenerator.poisson(N_electrons*I).astype(np.float64)
        return pos, I

    shift_std = np.empty([len(N_electrons), len(N_sampling_points)])
    width_std = np.empty_like(shift_std)

    shift = np.empty(n_rep)
    width = np.empty_like(shift)
    for p1 in tqdm(range(len(N_electrons))):
        for p2 in range(len(N_sampling_points)):
            Np = N_sampling_points[p2]
            Sp = N*S/Np
            A = np.empty(Np)
            for i in range(len(shift)):
                for ld in range(Np):
                    A[ld] = get_amplitude_at_single_position(
                        func, N_electrons[p1], Np, Sp*ld)

                shift[i], width[i] = fit_single_peak_spectrum(Sp, A)
            shift_std[p1, p2] = np.std(shift)
            width_std[p1, p2] = np.std(width)

    title = "Shift precision vs. Number of sampling points"
    fig = plot_simulated_data(N_sampling_points, N_electrons, 1e3*shift_std.T, add_linear_fit=False,
                              plot_title=title, xaxis_title="Number of sampling points", yaxis_title="Precision (MHz)",
                              legend_title="Number of<br>Brillouin<br>photoelectrons", legend_format='{:.1f}e-')
    fig.show()
    plotly_to_svg(fig, 'shift precision - sampling points.svg')

    # %% precision at different photoelectrons and number of sampling points (fixed number of photons)
    N_electrons = np.logspace(np.log10(5), 4, 5)
    N_sampling_points = np.round(np.logspace(
        np.log10(10), np.log10(1000), 30)).astype(np.int16)

    def func(pos, N_electrons, _):
        I = signal_single_peak(pos, shift_H2O, width_H2O)
        I = rndGenerator.poisson(N_electrons*I).astype(np.float64)
        return pos, I

    shift_std = np.empty([len(N_electrons), len(N_sampling_points)])
    width_std = np.empty_like(shift_std)

    shift = np.empty(n_rep)
    width = np.empty_like(shift)
    for p1 in tqdm(range(len(N_electrons))):
        for p2 in range(len(N_sampling_points)):
            Np = N_sampling_points[p2]
            Sp = N*S/Np
            A = np.empty(Np)
            for i in range(len(shift)):
                for ld in range(Np):
                    A[ld] = get_amplitude_at_single_position(
                        func, N*N_electrons[p1]/Np, Np, Sp*ld)

                shift[i], width[i] = fit_single_peak_spectrum(Sp, A)
            shift_std[p1, p2] = np.std(shift)
            width_std[p1, p2] = np.std(width)

    title = "Shift precision vs. Number of sampling points"
    fig = plot_simulated_data(N_sampling_points, N_electrons, 1e3*shift_std.T, add_linear_fit=False,
                              plot_title=title, xaxis_title="Number of sampling points", yaxis_title="Precision (MHz)",
                              legend_title="Number of<br>Brillouin<br>photoelectrons", legend_format='{:.1f}e-')
    fig.show()
    plotly_to_svg(fig, 'shift precision - sampling points (fixed photons).svg')

    #%% Combined contribution of different noise sources

    def simulate_precision_with_multiple_parameters(N_electrons, camera_noise,
                                                    stage_precision, Rayleigh_intensity,
                                                    intensity_noise, add_shot_noise=True):
        n_rep = 300

        shift = np.empty(n_rep)
        width = np.empty_like(shift)
        A = np.empty(N)
        for i in range(len(shift)):
            for ld in range(N):
                large_displ_um = S*ld
                pos = rndGenerator.normal(
                    large_displ_um, pos_err) + s*np.arange(n)

                # add noise sources
                # stage precision
                pos = rndGenerator.normal(pos, stage_precision)
                # Rayleigh intensity
                I = signal_single_peak(pos, shift_H2O, width_H2O,
                                       Rayleigh_intensity=Rayleigh_intensity)
                # intensity noise
                I = rndGenerator.normal(I, I*intensity_noise)
                I *= N_electrons
                # shot noise
                if add_shot_noise:
                    I = rndGenerator.poisson(I).astype(np.float64)
                # camera noise
                I = rndGenerator.normal(I+5*camera_noise, camera_noise)
                ####

                # fit a cosine
                pos_rel = pos - pos[0]
                popt, perr = fit_cos(pos_rel, I)

                A[ld] = popt[0]

                ref_phase = get_ref_phase(pos)
                ph_diff = np.angle(np.exp(1j*popt[2])*np.exp(-1j*ref_phase))
                if np.abs(ph_diff) > np.pi/2:
                    A[ld] *= -1

            shift[i], width[i] = fit_single_peak_spectrum(S, A)
        shift_std_MHz = 1e3*np.std(shift)
        width_std_MHz = 1e3*np.std(width)
        return shift_std_MHz, width_std_MHz

    N_electrons = 100
    camera_noise = 1.4  # e-
    stage_precision = 10e-3  # um
    Rayleigh_intensity = 1
    intensity_noise = 1e-2

    n_rep_std = 10
    shift_std = np.empty(n_rep_std)
    width_std = np.empty_like(shift_std)

    for i in range(n_rep_std):
        shift_std[i], width_std[i] = simulate_precision_with_multiple_parameters(N_electrons, camera_noise,
                                                                                 stage_precision, Rayleigh_intensity, intensity_noise)

    print(f"shift_std: {np.mean(shift_std):.2f}+/-{np.std(shift_std):.2f}MHz \
          width_std: {np.mean(width_std):.2f}+/-{np.std(width_std):.2f}MHz")
