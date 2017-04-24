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
    

