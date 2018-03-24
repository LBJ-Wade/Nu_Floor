"""

Define experimental parameters based on element

"""

import numpy as np
import os

path = os.getcwd()

mp = 0.931
year_to_sec = 3.154e7
joule_to_MeV = 6.242e12
miles_to_m = 1609.34
ft_to_m = 0.3048


def Element_Info(element):
    print element
    if element == 'Germanium':
        Qmin = 0.04
        Qmax = 50.
        Z = 32.
        Atope = np.array([70., 72., 73., 74., 76.])
        mfrac = np.array([0.212, 0.277, 0.077, 0.359, 0.074])
    
    elif element == 'Xenon':
        #Qmin = 1.0
        Qmin = 0.1
        Qmax = 50.
        Z = 54.
        Atope = np.array([128., 129., 130., 131., 132., 134., 136.])
        mfrac = np.array([0.019, 0.264, 0.041, 0.212, 0.269, 0.104, 0.089])

    elif element == 'Argon':
        Qmin = 1.
        Qmax = 50.
        Z = 18.
        Atope = np.array([40.])
        mfrac = np.array([1.])

    elif element == 'Sodium':
        Qmin = 1.
        Qmax = 50.
        Z = 11.
        Atope = np.array([23.])
        mfrac = np.array([1.])

    elif element == 'Iodine':
        Qmin = 1.
        Qmax = 50.
        Z = 53.
        Atope = np.array([127.])
        mfrac = np.array([1.])

    elif element == 'Fluorine':
        Qmin = 1.
        Qmax = 50.
        Z = 9.
        Atope = np.array([19.])
        mfrac = np.array([1.])

    elif element == 'FutureARGO':
        Qmin = 10.
        Qmax = 100.
        Z = 18.
        Atope = np.array([40.])
        mfrac = np.array([1.])
    elif element == 'FutureARGO_S2':
        Qmin = 0.6
        Qmax = 10.
        Z = 18.
        Atope = np.array([40.])
        mfrac = np.array([1.])
    elif element == 'FutureF':
        Qmin = 6. # check
        Qmax = 100.
        Z = 9.
        Atope = np.array([19.])
        mfrac = np.array([1.])
    elif element == 'FutureXe':
        Qmin = 1.
        Qmax = 100.
        Z = 54.
        Atope = np.array([128., 129., 130., 131., 132., 134., 136.])
        mfrac = np.array([0.019, 0.264, 0.041, 0.212, 0.269, 0.104, 0.089])
    elif element == 'FutureGe':
        Qmin = 0.04
        Qmax = 20.
        Z = 32.
        Atope = np.array([70., 72., 73., 74., 76.])
        mfrac = np.array([0.212, 0.277, 0.077, 0.359, 0.074])
    else:
        raise ValueError

    isotope = np.zeros((len(Atope), 4))
    for i in range(len(Atope)):
        isotope[i] = np.array([mp * Atope[i], Z, Atope[i], mfrac[i]])
    return isotope, Qmin, Qmax


def laboratory(elem, xen='LZ'):
    if elem == 'Germanium' or elem == 'Fluorine':
        lab = 'Snolab'
    elif elem == 'Xenon':
        if xen == 'LZ':
            lab = 'SURF'
        elif xen == 'X':
            lab = 'GS'
        else:
            raise ValueError
    elif elem == 'Argon':
        lab = 'GS'
    elif elem == 'Sodium' or elem == 'Iodine':
        lab = 'GS'
    else:
        lab = 'GS'
    return lab

# Name of reactor, Surface distance in km, Power output MWe
reactors_SURF = [['Cooper Nuclear Station', 801., 830.],
                 ['Monticello Nuclear Generating Plant', 788., 671],
                 ['Prarie Island Nuclear Generating Plant', 835., 1096.]]

reactors_SNOLAB = [['Nine Mile Point Nuclear Station U2', 498., 1140.],
                   ['Nine Mile Point Nuclear Station U1', 498., 621.],
                   ['R.E. Ginna Nuclear Power Plant', 468., 610.],
                   ['James A. Fitzpatrick Nuclear Power Plant', 500., 838.],
                   ['Point Beach Nuclear Power Plant U1', 552., 600.],
                   ['Point Beach Nuclear Power Plant U2', 552., 600.],
                   ['Fermi Unit II', 527., 1198.],
                   ['Davis-Besse Nuclear Power Station', 563., 889.],
                   ['Perry Nuclear Power Plant U1', 519., 1261.],
                   ['Bruce Nuclear Generating Station', 240., 6384.],
                   ['Darlington Nuclear Generating Station', 343., 3512.]]

reactors_GS = [['Tricastin 1-4', 744., 3820.],
               ['Cruas 1-4', 750., 3842.],
               ['St. Alban 1-2', 778., 2600.],
               ['Bugey 2-5', 760., 3724.]]

Nfiss = 6.
reac_runtime = 0.75
rntime_err = 0.06
Efiss = 205.3  # MeV


def reactor_flux(loc='Snolab'):
    if loc == 'Snolab':
        depth = 6000. * ft_to_m
        reactor_list = reactors_SNOLAB
    elif loc == 'SURF':
        depth = 8000. * ft_to_m
        reactor_list = reactors_SURF
    elif loc == 'GS':
        depth = 1.
        reactor_list = reactors_GS
    else:
        depth = 0.
        reactor_list = []

    flux = 0.
    err = 0.
    for reactor in reactor_list:
        flux += Nfiss * reactor[2] * joule_to_MeV * 1e6/Efiss * reac_runtime / \
                (4. * np.pi * (reactor[1]*1e5 + depth*1e2)**2.)
        err += Nfiss * reactor[2] * joule_to_MeV * 1e6/Efiss * reac_runtime / \
            (4. * np.pi * (reactor[1]*1e5 + depth*1e2)**2.)*np.sqrt((0.6/Efiss)**2. + (rntime_err/reac_runtime)**2. + (10./reactor[1])**2.)

    return flux, err

def geo_flux(loc='Snolab', el='U'):
    #print loc, el
    # element is either 'U' or 'Th' or 'K'
    if loc == 'Snolab':
        if el == 'U':
            flux = 4.9 * 10 ** 6.
            flx_err = 0.98 * 10 ** 6.
        elif el == 'Th':
            flux = 4.55 * 10 ** 6.
            flx_err = 1.17 * 10 ** 6.
        elif el == 'K':
            flux = 21.88 * 10 ** 6.
            flx_err = 3.67 * 10 ** 6.
        return flux, flx_err
    elif loc == 'SURF':
        if el == 'U':
            flux = 5.26 * 10**6.
            flx_err = 1.17 * 10**6.
        elif el == 'Th':
            flux = 4.90 * 10**6.
            flx_err = 1.34 * 10**6.
        elif el == 'K':
            flux = 22.68 * 10 ** 6.
            flx_err = 4.37 * 10 ** 6.
        return flux, flx_err
    elif loc == 'GS':
        if el == 'U':
            flux = 4.34 * 10**6.
            flx_err = 0.96 * 10 **6.
        elif el == 'Th':
            flux = 4.23 * 10**6.
            flx_err = 1.26 * 10 **6.
        elif el == 'K':
            flux = 20.54 * 10 ** 6.
            flx_err = 3.99 * 10 ** 6.
        return flux, flx_err

