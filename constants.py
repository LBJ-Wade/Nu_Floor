import numpy as np
from scipy.interpolate import interp1d
import os

path = os.getcwd()

YEAR_IN_S = 31557600.
GEV_IN_KEV = 1.e6
C_KMSEC = 299792.458

NUCLEON_MASS = 0.938272 # Nucleon mass in GeV
P_MAGMOM = 2.793 # proton magnetic moment, PDG Live
N_MAGMOM = -1.913 # neutron magnetic moment, PDG Live

NUCLEAR_MASSES = {
    'Xenon': 122.298654871,
    'Germanium': 67.663731424,
    'Argon': 37.2113263068,
    'Silicon': 26.1614775455,
    'Sodium': 21.4140502327,
    'Iodine': 118.206437626,
    'Fluorine': 17.6969003039,
    'He3': 3.,
    'Helium': 4.,
    'Nitrogen': 14.,
    'Neon': 20.,
    } # this is target nucleus mass in GeV: mT[GeV] = 0.9314941 * A[AMU]

ELEMENT_INFO = {"Xenon":{128:0.0192,129:0.2644,130:0.0408,131:0.2118,132:0.2689,134:0.1044,136:0.0887,'weight':131.1626},
                "Germanium":{70:0.2084,72:0.2754,73:0.0773,74:0.3628,76:0.0761,'weight':72.6905},
                "Iodine":{127:1.,'weight':127.},
                "Sodium":{23:1.,'weight':23.},
                "Silicon":{28:0.922,29:0.047,30:0.031,'weight':28.109},
                "Fluorine":{19:1.,'weight':19.},
                "Argon":{40:1.,'weight':40.},
                "Helium":{4:1.,'weight':4.},
                "He3":{3:1.,'weight':3.},
                "Nitrogen":{14:1.,'weight':14.},
                "Neon":{20:1.,'weight':20.}}

b8nu = np.loadtxt(path + '/Nu_Flux/B8NeutrinoFlux.dat')
b8nu_spectrum = interp1d(b8nu[:,0], b8nu[:,1], kind='cubic', fill_value=0., bounds_error=False)

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
geo_u = np.loadtxt(path + '/Nu_Flux/GeoU.dat')
geo_th = np.loadtxt(path + '/Nu_Flux/GeoTh.dat')
geo_k = np.loadtxt(path + '/Nu_Flux/GeoK.dat')
geoU_spectrum = interp1d(geo_u[:,0], geo_u[:,1], kind='linear', fill_value=0., bounds_error=False)
geoTh_spectrum = interp1d(geo_th[:,0], geo_th[:,1], kind='linear', fill_value=0., bounds_error=False)
geoK_spectrum = interp1d(geo_k[:,0], geo_k[:,1], kind='linear', fill_value=0., bounds_error=False)


def atm_spectrum(x):
    return atmnue_spectrum(x) + atmnumu_spectrum(x) + atmnumubar_spectrum(x) + atmnuebar_spectrum(x)

NEUTRINO_EMAX = {"b8": 16.18,
                 "b7l1": 0.39,
                 "b7l2": 0.87,
                 "pepl1": 1.45,
                 "hep": 18.77,
                 "pp": 0.42,
                 "o15": 1.73,
                 "n13": 1.20,
                 "f17": 1.74,
                 "atmnue": 9.44*10**2.,
                 "atmnuebar": 9.44*10**2.,
                 "atmnumu": 9.44*10**2.,
                 "atmnumubar": 9.44*10**2.,
                 "dsnb3mev":36.90,
                 "dsnb5mev": 57.01,
                 "dsnb8mev": 81.91,
                 "reactor": 10.,
                 "geoU": 3.99,
                 "geoTh": 2.26,
                 "geoK": 1.32,
                 "atm": 9.44*10**2.
                 }


NEUTRINO_MEANF = {"b8": 5.58 * 10. ** 6.,
                  "b7l1": 0.1 * 5.00 * 10. ** 9.,
                  "b7l2": 0.9 * 5.00 * 10. ** 9.,
                  "pepl1": 1.44 * 10. ** 8.,
                  "hep": 8.04 * 10. ** 3.,
                  "pp": 5.98 * 10. ** 10.,
                  "o15": 2.23 * 10. ** 8.,
                  "n13": 2.96 * 10. ** 8.,
                  "f17": 5.52 * 10. ** 6.,
                  "atmnue": 1.27 * 10. ** 1,
                  "atmnuebar": 1.17 * 10. ** 1,
                  "atmnumu": 2.46 * 10. ** 1,
                  "atmnumubar": 2.45 * 10. ** 1,
                  "dsnb3mev": 4.55 * 10. ** 1,
                  "dsnb5mev": 2.73 * 10. ** 1,
                  "dsnb8mev": 1.75 * 10. ** 1,
                  "atm": (1.27 + 1.17 + 2.46 + 2.45) * 10.
                  }

NEUTRINO_SIG = {"b8": 5.58 * 10. ** 6. * 0.14,
                  "b7l1": 0.07 * 0.1 * 5.00 * 10. ** 9.,
                  "b7l2": 0.07 * 0.9 * 5.00 * 10. ** 9.,
                  "pepl1": 0.012 * 1.44 * 10. ** 8.,
                  "hep": 0.3 * 8.04 * 10. ** 3.,
                  "pp": 0.006 * 5.98 * 10. ** 10.,
                  "o15": 0.15 * 2.23 * 10. ** 8.,
                  "n13": 0.14 * 2.96 * 10. ** 8.,
                  "f17": 0.17 * 5.52 * 10. ** 6.,
                  "atmnue": 0.5 * 1.27 * 10. ** 1,
                  "atmnuebar": 0.5 * 1.17 * 10. ** 1,
                  "atmnumu": 0.5 * 2.46 * 10. ** 1,
                  "atmnumubar": 0.5 * 2.45 * 10. ** 1,
                  "dsnb3mev": 0.5 * 4.55 * 10. ** 1,
                  "dsnb5mev": 0.5 * 2.73 * 10. ** 1,
                  "dsnb8mev": 0.5 * 1.75 * 10. ** 1,
                  "atm": 0.5 * (1.27 + 1.17 + 2.46 +2.45) * 10.
                  }
NEUTRINO_SPEC = {"b8": b8nu_spectrum,
                  "hep": hepnu_spectrum,
                  "pp": ppnu_spectrum,
                  "o15": o15nu_spectrum,
                  "n13": n13nu_spectrum,
                  "f17": f17nu_spectrum,
                  "atmnue": atmnue_spectrum,
                  "atmnuebar": atmnuebar_spectrum,
                  "atmnumu": atmnumu_spectrum,
                  "atmnumubar": atmnumubar_spectrum,
                  "dsnb3mev": dsnb3mevnu_spectrum,
                  "dsnb5mev": dsnb5mevnu_spectrum,
                  "dsnb8mev": dsnb8mevnu_spectrum,
                  "reactor": reactor_nu_spectrum,
                  "geoU": geoU_spectrum,
                  "geoTh": geoTh_spectrum,
                  "geoK": geoK_spectrum,
                  "atm": atm_spectrum
                  }

nu_lines = ['b7l1', 'b7l2', 'pepl1']
line_flux = [(0.1) * 5.00 * 10. ** 9., (0.9) * 5.00 * 10. ** 9., 1.44 * 10. ** 8.]
e_lines = [0.380, 0.860, 1.440]


