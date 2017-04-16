"""

Define experimental parameters based on element

"""

import numpy as np

mp = 0.931

def Element_Info(element):

    if element == 'germanium':
        Qmin = 0.1
        Qmax = 7.
        Z = 32.
        Atope = np.array([70., 72., 73., 74., 76.])
        mfrac = np.array([0.212, 0.277, 0.077, 0.359, 0.074])
    
    elif element == 'xenon':
        Qmin = 0.1
        Qmax = 10.
        Z = 54.
        Atope = np.array([128., 129., 130., 131., 132., 134., 136.])
        mfrac = np.array([0.019, 0.264, 0.041, 0.212, 0.269, 0.104, 0.089])

    elif element == 'argon':
        Qmin = 0.1
        Qmax = 50.
        Z = 18.
        Atope = np.array([40.])
        mfrac = np.array([1.])

    elif element == 'silicon':
        Qmin = 0.1
        Qmax = 50.
        Z = 14.
        Atope = np.array([28., 29., 30.])
        mfrac = np.array([0.922, 0.047, 0.031])

    elif element == 'fluorine':
        Qmin = 0.1
        Qmax = 50.
        Z = 9.
        Atope = np.array([19.])
        mfrac = np.array([1.])

    else:
        raise ValueError

    isotope = np.zeros((len(Atope), 4))
    for i in range(len(Atope)):
        isotope[i] = np.array([mp * Atope[i], Z, Atope[i], mfrac[i]])
    return isotope, Qmin, Qmax



