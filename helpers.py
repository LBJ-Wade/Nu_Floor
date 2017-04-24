import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.optimize import brentq, curve_fit
from scipy.stats import norm
import numpy.random as random
#from scipy.interpolate import griddata,interp1d,interp2d

path = os.getcwd()

pi = np.pi #3.14159265359
c_lgt = 2.998 * 10.**5.

#eta0_a0_tabbed = np.loadtxt(os.environ['DMDD_AM_MAIN_PATH']+'/..'+'/dmdd/eta0_a0.dat')
#cdef np.float_t[:,:] eta0_a0_tabbed = np.loadtxt(os.environ['DMDD_AM_MAIN_PATH']+'/..'+'/dmdd/eta0_a0.dat')
#eta0_a1_tabbed = np.loadtxt(os.environ['DMDD_AM_MAIN_PATH']+'/..'+'/dmdd/eta0_a1.dat')
#cdef np.float_t[:,:] eta0_a1_tabbed = np.loadtxt(os.environ['DMDD_AM_MAIN_PATH']+'/..'+'/dmdd/eta0_a1.dat')
#eta0_b1_tabbed = np.loadtxt(os.environ['DMDD_AM_MAIN_PATH']+'/..'+'/dmdd/eta0_b1.dat')
#cdef np.float_t[:,:] eta0_b1_tabbed = np.loadtxt(os.environ['DMDD_AM_MAIN_PATH']+'/..'+'/dmdd/eta0_b1.dat')
#eta1_a0_tabbed = np.loadtxt(os.environ['DMDD_AM_MAIN_PATH']+'/../'+'/dmdd/eta1_a0.dat')
#cdef np.float_t[:,:] eta1_a0_tabbed = np.loadtxt(os.environ['DMDD_AM_MAIN_PATH']+'/../'+'/dmdd/eta1_a0.dat')
#eta1_a1_tabbed = np.loadtxt(os.environ['DMDD_AM_MAIN_PATH']+'/../'+'/dmdd/eta1_a1.dat')
#cdef np.float_t[:,:] eta1_a1_tabbed = np.loadtxt(os.environ['DMDD_AM_MAIN_PATH']+'/../'+'/dmdd/eta1_a1.dat')
#eta1_b1_tabbed = np.loadtxt(os.environ['DMDD_AM_MAIN_PATH']+'/../'+'/dmdd/eta1_b1.dat')
#cdef np.float_t[:,:] eta1_b1_tabbed = np.loadtxt(os.environ['DMDD_AM_MAIN_PATH']+'/../'+'/dmdd/eta1_b1.dat')

b8nu = np.loadtxt(path + '/Nu_Flux/B8NeutrinoFlux.dat')
b8nu_spectrum = interp1d(b8nu[:,0], b8nu[:,1], kind='cubic', fill_value=0., bounds_error=False)

#note lines have linear interpolation in bin

b7nul1 = np.loadtxt(path + '/Nu_Flux/B7NeutrinoLine1.dat')
b7nul1_spectrum = interp1d(b7nul1[:,0], b7nul1[:,1], kind='linear', fill_value=0., bounds_error=False)

b7nul2 = np.loadtxt(path + '/Nu_Flux/B7NeutrinoLine2.dat')
b7nul2_spectrum = interp1d(b7nul2[:,0], b7nul2[:,1], kind='linear', fill_value=0., bounds_error=False)

pepnul1 = np.loadtxt(path + '/Nu_Flux/PEPNeutrinoLine1.dat')
pepnul1_spectrum = interp1d(pepnul1[:,0], pepnul1[:,1], kind='linear', fill_value=0., bounds_error=False)

hepnu = np.loadtxt(path + '/Nu_Flux/HEPNeutrinoFlux.dat')
hepnu_spectrum = interp1d(hepnu[:,0], hepnu[:,1], kind='cubic', fill_value=0., bounds_error=False)

ppnu = np.loadtxt(path + '/Nu_Flux/PPNeutrinoFlux.dat')
ppnu_spectrum = interp1d(ppnu[:,0], ppnu[:,1], kind='cubic', fill_value=0., bounds_error=False)

o15nu = np.loadtxt(path + '/Nu_Flux/O15NeutrinoFlux.dat')
o15nu_spectrum = interp1d(o15nu[:,0], o15nu[:,1], kind='cubic', fill_value=0., bounds_error=False)

n13nu = np.loadtxt(path + '/Nu_Flux/N13NeutrinoFlux.dat')
n13nu_spectrum = interp1d(n13nu[:,0], n13nu[:,1], kind='cubic', fill_value=0., bounds_error=False)

f17nu = np.loadtxt(path + '/Nu_Flux/F17NeutrinoFlux.dat')
f17nu_spectrum = interp1d(f17nu[:,0], f17nu[:,1], kind='cubic', fill_value=0., bounds_error=False)

atmnue = np.loadtxt(path + '/Nu_Flux/atmnue_noosc_fluka_flux_norm.dat')
atmnue_spectrum = interp1d(atmnue[:,0], atmnue[:,1], kind='cubic', fill_value=0., bounds_error=False)

atmnuebar = np.loadtxt(path + '/Nu_Flux/atmnuebar_noosc_fluka_flux_norm.dat')
atmnuebar_spectrum = interp1d(atmnuebar[:,0], atmnuebar[:,1], kind='cubic', fill_value=0., bounds_error=False)

atmnumu = np.loadtxt(path + '/Nu_Flux/atmnumu_noosc_fluka_flux_norm.dat')
atmnumu_spectrum = interp1d(atmnumu[:,0], atmnumu[:,1], kind='cubic', fill_value=0., bounds_error=False)

atmnumubar = np.loadtxt(path + '/Nu_Flux/atmnumubar_noosc_fluka_flux_norm.dat')
atmnumubar_spectrum = interp1d(atmnumubar[:,0], atmnumubar[:,1], kind='cubic', fill_value=0., bounds_error=False)

dsnb3mevnu = np.loadtxt(path + '/Nu_Flux/dsnb_3mev_flux_norm.dat')
dsnb3mevnu_spectrum = interp1d(dsnb3mevnu[:,0], dsnb3mevnu[:,1], kind='cubic', fill_value=0., bounds_error=False)

dsnb5mevnu = np.loadtxt(path + '/Nu_Flux/dsnb_5mev_flux_norm.dat')
dsnb5mevnu_spectrum = interp1d(dsnb5mevnu[:,0], dsnb5mevnu[:,1], kind='cubic', fill_value=0., bounds_error=False)

dsnb8mevnu = np.loadtxt(path + '/Nu_Flux/dsnb_8mev_flux_norm.dat')
dsnb8mevnu_spectrum = interp1d(dsnb8mevnu[:,0], dsnb8mevnu[:,1], kind='cubic', fill_value=0., bounds_error=False)

# Reactor Nus
reactor_nu = np.loadtxt(path + '/Nu_Flux/Reactor_Spectrum.dat')
reactor_nu_spectrum = interp1d(reactor_nu[:,0], reactor_nu[:,1], kind='cubic', fill_value=0., bounds_error=False)

# Geo nus
geo_nu = np.loadtxt(path + '/Nu_Flux/Geo_nu.dat')
geoU_spectrum = interp1d(geo_nu[:,0], geo_nu[:,1], kind='linear', fill_value=0., bounds_error=False)
geoTh_spectrum = interp1d(geo_nu[:,0], geo_nu[:,2], kind='linear', fill_value=0., bounds_error=False)


default_rate_parameters = dict(mass=50., sigma_si=0., sigma_sd=0., sigma_anapole=0., sigma_magdip=0.,
                                sigma_elecdip=0., sigma_LS=0., sigma_f1=0., sigma_f2=0., sigma_f3=0.,
                                sigma_si_massless=0., sigma_sd_massless=0., sigma_anapole_massless=0.,
                                sigma_magdip_massless=0.,  sigma_elecdip_massless=0., sigma_LS_massless=0.,
                                sigma_f1_massless=0.,  sigma_f2_massless=0.,  sigma_f3_massless=0.,
                                fnfp_si=1.,  fnfp_sd=1., fnfp_anapole=1.,  fnfp_magdip=1.,  fnfp_elecdip=1.,
                                fnfp_LS=1.,  fnfp_f1=1.,  fnfp_f2=1.,  fnfp_f3=1., fnfp_si_massless=1.,
                                fnfp_sd_massless=1., fnfp_anapole_massless=1.,  fnfp_magdip_massless=1.,
                                fnfp_elecdip_massless=1., fnfp_LS_massless=1.,  fnfp_f1_massless=1.,
                                fnfp_f2_massless=1.,  fnfp_f3_massless=1., v_lag=220.,  v_rms=220.,
                                v_esc=533.,  rho_x=0.3, delta=0., GF=False, time_info=False)

def v_leq_vesc(delta, mT, vesc=533.+232.):
    delta *= 1e-6
    vesc /= c_lgt
    return 2. * delta / (vesc**2. * (1. - 2.*delta/(mT * vesc**2.)))

def v_delta(delta, mT, mx):
    if delta <= 0:
        return 0.
    else:
        return np.sqrt(2. * delta*1e-6 * (mx + mT) / (mx * mT))*c_lgt

def ERplus(mx, mT, v, delta):
    r_mass = mx * mT / (mx + mT)
    if delta > 1e6 * (v/c_lgt)**2. * r_mass/2.:
        return 0.
    sqt_t = np.sqrt(1. - 2. * delta * 1e-6 * c_lgt**2. / (r_mass * v**2.))
    return (v/c_lgt)**2. * r_mass**2. /(2.*mT) * 1e6 * (1. + sqt_t)**2.

def ERminus(mx, mT, v, delta):
    r_mass = mx * mT / (mx + mT)
    if delta > 1e6 * (v/c_lgt)**2. * r_mass/2.:
        return 0.
    sqt_t = np.sqrt(1. - 2. * delta * 1e-6 * c_lgt**2. / (r_mass * v**2.))
    return (v/c_lgt)**2. * r_mass**2. /(2.*mT) * 1e6 * (1. - sqt_t)**2.


def mxRange(delta, mT, Emin, Emax, vesc=533.+232.):
    kG = 1e-6
    Emax *= kG
    Emin *= kG
    delta *= kG
    vesc /= c_lgt
    mx_p1 = (-Emax**2.*mT + np.sqrt(2.)*(Emax*mT)**3./2.*vesc - Emax*mT*delta) / \
           (Emax**2. - 2.*Emax*mT*vesc**2.+2.*Emax*delta+delta**2.)
    mx_p2 = (-Emin ** 2. * mT + np.sqrt(2.) * (Emin * mT) ** 3. / 2. * vesc - Emin * mT * delta) / \
           (Emin ** 2. - 2. * Emin * mT * vesc ** 2. + 2. * Emin * delta + delta ** 2.)
    m_range = np.sort(np.concatenate((mx_p1, mx_p2)))
    return np.min(m_range), np.max(m_range)


def MinDMMass(mT, delta, eng, vesc=533.+232.):
    if delta <= 0:
        mmin = 0.001
    else:
        term = 2. * (c_lgt/vesc)**2. * delta * 1e-6 / mT
        mmin = mT * term / (1. - term)

    solve = brentq(lambda x: ERplus(x, mT, vesc, delta) - eng, mmin, 1000.)
    return solve


def trapz(y, x):
    """
    Integrator function, the same as numpy.trapz.
    The points y,x must be given in increasing order.
    """
    if len(x) != len(y):
        raise ValueError('x and y must be same length')
    npts = len(x)
    tot = 0
    
    for i in range(npts-1):
        tot += 0.5*(y[i]+y[i+1])*(x[i+1]-x[i])
    return tot



def log_fact(xx):
    """
    Approximate formula for ln(x!).
    """
    ser=1.000000000190015
    cof = [76.18009172947146,-86.50532032941677,
           24.01409824083091,-1.231739572450155,
           0.1208650973866179e-2,-0.5395239384953e-5]
    
    y=xx+1
    x=xx+1
    tmp=x+5.5
    tmp -= (x+0.5)*log(tmp)
    for c in cof:
        y=y+1
        ser += c/y
    return -tmp+log(2.5066282746310005*ser/x)


def eta(v_min, v_esc, v_rms, v_lag):
    """
    This is the correctly scaled velocity integral for a rate
    with no special velocity dependence.

    The input units for all velocities are km/s.
    """
    
    xmin = v_min/v_rms
    xe = v_lag/v_rms + 17.72/v_rms
    xesc = v_esc/v_rms
    norm = 1./(4*xe*(np.sqrt(pi)*0.5*erf(xesc)-np.exp(-xesc**2)*xesc*(1+2*xesc**2/3.)))

    if xesc - xe - xmin > 0:
        integral = (np.sqrt(pi)*(erf(xe+xmin)-erf(xmin-xe))-4*xe*np.exp(-xesc**2)*(1+xesc**2-xmin**2-xe**2./3))*norm
    elif xesc - xmin + xe > 0 and -xesc+xmin+xe>0:
        integral =  (np.sqrt(pi)*(erf(xesc)-erf(xmin-xe))-2*np.exp(-xesc**2)*(xesc+xe-xmin-1./3*(xe-2*xesc-xmin)*(xesc+xe-xmin)**2))*norm
    else:
        integral =  0
    
    res = 3*10**5*integral/v_rms
    
    return res


def eta_GF(v_min, time, time_info):
    """
    Calculated only for v_esc = 533km/s, v_rms = v_lag = 220 km/s
    
    Also, if vmin_
    """
    
    vmin_max=700.
    vlag = 220.
    mod_amp = 29.8 * 0.49
    mod_phase = 0.42
    twopi = 2. * pi


    global eta0_a0_tabbed
    global eta0_a1_tabbed
    global eta0_b1_tabbed
    
    a0_x = eta0_a0_tabbed[:,0]
    a0_y = eta0_a0_tabbed[:,1]
    a1_x = eta0_a1_tabbed[:,0]
    a1_y = eta0_a1_tabbed[:,1]
    b1_x = eta0_b1_tabbed[:,0]
    b1_y = eta0_b1_tabbed[:,1]
    

    if v_min <= vmin_max:
        if time_info:
            res = 3. * 10.**5. * (interp1d(a0_x, a0_y, v_min) + 
                  interp1d(a1_x, a1_y, v_min) * cos(2. * pi * (time - 0.4178)) +
                  interp1d(b1_x, b1_y, v_min) * sin(2. * pi * (time - 0.4178))) 
        else:   
            res = 3. * 10.**5. * (interp1d(a0_x, a0_y, v_min))
    else:
        if time_info:
            vlag += mod_amp * cos(twopi * (time - mod_phase))
        res = eta(v_min, 533., 220., vlag)
    
    return res

def zeta_GF(v_min, time, time_info):
    """
    This is the correctly scaled velocity integral for a rate
    with no special velocity dependence.

    The input units for all velocities are km/s.
    """
    
    vmin_max=700.
    vlag = 220.
    mod_amp = 29.8 * 0.49
    mod_phase = 0.42
    twopi = 2. * pi

    global eta1_a0_tabbed
    global eta1_a1_tabbed
    global eta1_b1_tabbed
    
    a0_x = eta1_a0_tabbed[:,0]
    a0_y = eta1_a0_tabbed[:,1]
    a1_x = eta1_a1_tabbed[:,0]
    a1_y = eta1_a1_tabbed[:,1]
    b1_x = eta1_b1_tabbed[:,0]
    b1_y = eta1_b1_tabbed[:,1]
    

    if v_min <= vmin_max:
        if time_info:
            res = (interp1d(a0_x, a0_y, v_min) + 
                   interp1d(a1_x, a1_y, v_min) * cos(2. * pi * (time - 0.4178)) +
                   interp1d(b1_x, b1_y, v_min) * sin(2. * pi * (time - 0.4178)))  / (3.*10**5.) #+
        else:
            res = interp1d(a0_x, a0_y, v_min) / (3.*10**5.) 
    else:
        if time_info:
            vlag += mod_amp * cos(twopi * (time - mod_phase))
        res = zeta(v_min, 533., 220., vlag)

    return res


def zeta(v_min, v_esc, v_rms, v_lag):
    """
    This is the correctly scaled velocity integral for a rate with additional velocity^2 dependence. The input units for all velocities are km/s.
    """
    
    xmin = v_min/v_rms
    xe = v_lag/v_rms + 17.72/v_rms
    xesc = v_esc/v_rms
    norm = 1./(4*xe*(np.sqrt(pi)*0.5*erf(xesc)-np.exp(-xesc**2)*xesc*(1+2*xesc**2/3.)))
    
    if xesc-xmin-xe > 0:
        integral = (1./2*np.sqrt(pi)*(2*xe**2+1)*(erf(xmin+xe)-erf(xmin-xe)) + np.exp(-(xmin-xe)**2)*(xe+xmin) + np.exp(-(xmin+xe)**2)*(xe-xmin) - np.exp(-xesc**2)*(4*(1+xesc**2+xe**2./3)*xe-2*xe**5/15.+4*xe**3*xesc**2/3.-2*xe*(xmin**4-xesc**4)))*norm
    elif xesc-xmin+xe > 0 and -xesc+xmin+xe > 0:
        integral = 2*np.exp(-xesc**2)*(1./3*(xmin**3-(xesc+xe)**3)-1./4*np.exp(-xmin**2-xe**2+xesc**2)*(pi**.5*(2*xe**2+1)*np.exp(xmin**2+xe**2)*(erf(xmin-xe)-erf(xesc))-2*np.exp(2*xmin*xe)*(xmin+xe)+2*(xesc+2*xe)*np.exp(xmin**2+xe**2-xesc**2))-1./30*(-xe**5+10*xe**3*xesc**2+10*xe**2*(2*xesc**3+xmin**3)+15*xe*(xesc**4-xmin**4)+4*xesc**5-10*xesc**2*xmin**3+6*xmin**5))*norm
    else:
        integral =  0
    res = integral*v_rms/(3.*10**5.)
    return res


def adaptive_samples(sig_min, sig_max, list):
    arr_l = np.asarray(list)

    if len(list) == 0:
        return np.mean(np.array([sig_min, sig_max])), False
    if len(list) == 1:
        if list[0][1] < 0.9:
            return np.mean(np.array([list[0][0], sig_max])), False
        else:
            return np.mean(np.array([list[0][0], sig_min])), False
    if np.sum((arr_l[:, 1] > 0.15) & (arr_l[:, 1] < 0.95)) > 5:
        print 'Enough points in [0.5, 0.95]'
        print arr_l
        return sig_min, True
    high = np.sum(arr_l[:, 1] > 0.9)
    low = np.sum(arr_l[:, 1] < 0.9)
    #print 'High, Low', high, low
    if low == 0:
        return np.mean(np.array(np.min(arr_l[:, 0]), sig_min)), False
    elif high == 0:
        return np.mean(np.array(np.max(arr_l[:, 0]), sig_max)), False
    else:
        mean = sum(arr_l[:,0] * arr_l[:,1]) / sum(arr_l[:,1])
        popt, pcov = curve_fit(gauss_cdf_function, arr_l[:,0], arr_l[:,1], p0=[mean, 1.])
        xrag = np.linspace(sig_min, sig_max, 300)
        ypts = gauss_cdf_function(xrag, *popt)
        ypts /= ypts.max()
        u = random.rand(1)
        sig = xrag[np.absolute(ypts - u).argmin()]
    return sig, False


def gauss_cdf_function(x, x0, sigma):
    z = (x - x0) / (np.sqrt(2.) * sigma)
    return (np.pi / 2.) * sigma * (erf(z) + 1.)

