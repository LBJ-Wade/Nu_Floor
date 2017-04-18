"""

Code info:

"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import romberg
from rate_UV import *
from helpers import *
import os
from experiments import *

path = os.getcwd()

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



gF = 1.16637 * 10. ** -5. # Gfermi in GeV^-2
sw = 0.2312 # sin(theat_weak)^2
MeVtofm = 0.0050677312
s_to_yr = 3.154*10.**7.


class Likelihood_analysis(object):

    def __init__(self, model, coupling, mass, dm_sigp, fnfp, exposure, element, isotopes,
                 energies, times, nu_names, lab, nu_spec, Qmin, Qmax, time_info=False, GF=False):

        nu_resp = np.zeros(nu_spec, dtype=object)

        self.nu_resp = np.zeros(nu_spec, dtype=object)
        self.nu_int_resp = np.zeros(nu_spec, dtype=object)

        self.events = energies
        self.element = element
        self.mass = mass
        self.dm_sigp = dm_sigp
        self.exposure = exposure

        self.Qmin = Qmin
        self.Qmax = Qmax

        self.nu_spec = nu_spec
        self.nu_names = nu_names
        self.lab = lab

        self.times = times

        like_params =default_rate_parameters.copy()

        like_params['element'] = element
        like_params['mass'] = mass
        like_params[model] = dm_sigp
        like_params[coupling] = fnfp

        like_params['GF'] = GF
        like_params['time_info'] = time_info

        # TODO: Right now im cutting corners. All sigmas and fnfps need to be passed directly into this function
        # and values need to be controled in main.py
        self.dm_recoils = dRdQ(energies, times, **like_params) * 10.**3. * s_to_yr

        self.dm_integ = R(Qmin=self.Qmin, Qmax=self.Qmax, **like_params) * 10.**3. * s_to_yr

        eng_lge = np.logspace(np.log10(self.Qmin), np.log10(self.Qmax), 400)

        for i in range(nu_spec):
            nu_resp[i] = np.zeros_like(eng_lge)

        for iso in isotopes:
            # isotope list [mT, Z, A, frac]
            for j in range(nu_spec):
                nu_resp[j] += Nu_spec(self.lab).nu_rate(nu_names[j], eng_lge, iso)

        for i in range(nu_spec):
            #linear interpolation for the 3 lines
            if i == 1 or i == 2 or i == 3:
                self.nu_resp[i] = interp1d(eng_lge, nu_resp[i], kind='linear', bounds_error=False, fill_value=0.)
                self.nu_int_resp[i] = np.trapz(self.nu_resp[i](eng_lge), eng_lge)
            else:
                self.nu_resp[i] = interp1d(eng_lge, nu_resp[i], kind='cubic', bounds_error=False, fill_value=0.)
                self.nu_int_resp[i] = np.trapz(self.nu_resp[i](eng_lge), eng_lge)

                #def like_multi_grad(self, norms):
                # Not used at moment
                #nu_norm = norms[0]
                #sig_dm = norms[1]
                #return self.likelihood(nu_norm, sig_dm, return_grad=True)

    def like_multi_wrapper(self, norms):
        nu_norm = np.zeros(self.nu_spec, dtype=object)

        for i in range(self.nu_spec):
            nu_norm[i] = norms[i]

        sig_dm = norms[-1]
        return self.likelihood(nu_norm, sig_dm, return_grad=False)

    def likelihood(self, nu_norm, sig_dm, return_grad=False):
        # TODO: consider implimenting gradient if minimization has difficulties
        # - 2 log likelihood
        # nu_norm in units of cm^-2 s^-1, sig_dm in units of cm^2
        like = 0.
        grad_x = 0.
        diff_nu = np.zeros(self.nu_spec, dtype=object)
        grad_nu = np.zeros(self.nu_spec)
        nu_events = np.zeros(self.nu_spec, dtype=object)

        n_obs = len(self.events)
        # total rate contribution

        dm_events = 10. ** sig_dm * self.dm_integ * self.exposure

        for i in range(self.nu_spec):
            nu_events[i] = 10. ** nu_norm[i] * self.exposure * self.nu_int_resp[i]

        like += 2. * (dm_events + sum(nu_events))

        #grad_x += 2. * np.log(10.) * dm_events
        #grad_nu[1] += 2. * np.log(10.) * nu_events[1]
        #grad_nu[2] += 2. * np.log(10.) * nu_events[2]
        #print 'Nobs: , b8 events: , DM events: ',n_obs, b8_events, dm_events

        # Differential contribution
        diff_dm = self.dm_recoils * self.exposure

        for i in range(self.nu_spec):
            diff_nu[i] = self.nu_resp[i](self.events) * self.exposure

        lg_vle = (10. ** sig_dm * diff_dm + np.dot(list(map(lambda x:10**x,nu_norm)),diff_nu)) #nu norm array
        for i in range(len(lg_vle)):
            if lg_vle[i] > 0.:
                like += -2. * np.log(lg_vle[i])
        #print lg_vle.sum()
        #grad_x += -2.*np.log(10.)*dm_events / lg_vle.sum()
        #grad_nu += -2. * np.log(10.) * b8_events / lg_vle.sum()

        # nu normalization contribution
        for i in range(self.nu_spec):
            like += self.nu_gaussian(self.nu_names[i], nu_norm[i])

        #+ self.nu_gaussian(nu_norm2) ...

        #grad_nu += self.nu_gaussian(b8_norm, return_deriv=True)

        #if return_grad:
        #    return [grad_nu, grad_x]
        #else:
        return like


    def nu_gaussian(self, nu_component, flux_n, return_deriv=False):
        # - 2 log of gaussian flux norm comp

        b8_mean_f = 5.58 * 10. ** 6. 		 # cm^-2 s^-1
        b8_sig = b8_mean_f * (0.14)     	 # cm^-2 s^-1

        b7l1_mean_f = (0.1) * 5.00 * 10. ** 9.
        b7l1_sig = b7l1_mean_f * (0.07)

        b7l2_mean_f = (0.9) * 5.00 * 10. ** 9.
        b7l2_sig = b7l2_mean_f * (0.07)

        pepl1_mean_f = 1.44 * 10. ** 8.
        pepl1_sig = pepl1_mean_f * (0.012)

        hep_mean_f = 8.04 * 10. ** 3.
        hep_sig = hep_mean_f * (0.3)

        pp_mean_f = 5.98 * 10. ** 10.
        pp_sig = pp_mean_f * (0.006)

        o15_mean_f = 2.23 * 10. ** 8.
        o15_sig = o15_mean_f * (0.15)

        n13_mean_f = 2.96 * 10. ** 8.
        n13_sig = n13_mean_f * (0.14)

        f17_mean_f = 5.52 * 10. ** 6.
        f17_sig = f17_mean_f * (0.17)

        atmnue_mean_f = 1.27 * 10. ** 1
        atmnue_sig = atmnue_mean_f * (0.5)		 	# take 50% error

        atmnuebar_mean_f = 1.17 * 10. ** 1
        atmnuebar_sig = atmnuebar_mean_f * (0.5)		# take 50% error

        atmnumu_mean_f = 2.46 * 10. ** 1
        atmnumu_sig = atmnumu_mean_f * (0.5)    		# take 50% error

        atmnumubar_mean_f = 2.45 * 10. ** 1
        atmnumubar_sig = atmnumubar_mean_f * (0.5)    	# take 50% error

        dsnb3mev_mean_f = 4.55 * 10. ** 1
        dsnb3mev_sig = dsnb3mev_mean_f * (0.5) 			# take 50% error

        dsnb5mev_mean_f = 2.73 * 10. ** 1
        dsnb5mev_sig = dsnb5mev_mean_f * (0.5)  		# take 50% error

        dsnb8mev_mean_f = 1.75 * 10. ** 1
        dsnb8mev_sig = dsnb8mev_mean_f * (0.5)			# take 50% error


        reactor_mean_f, reactor_sig = reactor_flux(loc=self.lab)

        geoU_mean_f, geoU_sig = geo_flux(loc=self.lab, el='U')
        geoT_mean_f, geoT_sig = geo_flux(loc=self.lab, el='Th')

        if nu_component == 'b8':
            return b8_mean_f**2. * (10. ** flux_n - 1.)**2. / b8_sig**2.
        elif nu_component == 'b7l1':
            return b7l1_mean_f**2. * (10. ** flux_n - 1.)**2. / b7l1_sig**2.
        elif nu_component == 'b7l2':
            return b7l2_mean_f**2. * (10. ** flux_n - 1.)**2. / b7l2_sig**2.
        elif nu_component == 'pepl1':
            return pepl1_mean_f**2. * (10. ** flux_n - 1.)**2. / pepl1_sig**2.
        elif nu_component == 'hep':
            return hep_mean_f**2. * (10. ** flux_n - 1.)**2. / hep_sig**2.
        elif nu_component == 'pp':
            return pp_mean_f**2. * (10. ** flux_n - 1.)**2. / pp_sig**2.
        elif nu_component == 'o15':
            return o15_mean_f**2. * (10. ** flux_n - 1.)**2. / o15_sig**2.
        elif nu_component == 'n13':
            return n13_mean_f**2. * (10. ** flux_n - 1.)**2. / n13_sig**2.
        elif nu_component == 'f17':
            return f17_mean_f**2. * (10. ** flux_n - 1.)**2. / f17_sig**2.
        elif nu_component == 'atmnue':
            return atmnue_mean_f**2. * (10. ** flux_n - 1.)**2. / atmnue_sig**2.
        elif nu_component == 'atmnuebar':
            return atmnuebar_mean_f**2. * (10. ** flux_n - 1.)**2. / atmnuebar_sig**2.
        elif nu_component == 'atmnumu':
            return atmnumu_mean_f**2. * (10. ** flux_n - 1.)**2. / atmnumu_sig**2.
        elif nu_component == 'atmnumubar':
            return atmnumubar_mean_f**2. * (10. ** flux_n - 1.)**2. / atmnumubar_sig**2.
        elif nu_component == 'dsnb3mev':
            return dsnb3mev_mean_f**2. * (10. ** flux_n - 1.)**2. / dsnb3mev_sig**2.
        elif nu_component == 'dsnb5mev':
            return dsnb5mev_mean_f**2. * (10. ** flux_n - 1.)**2. / dsnb5mev_sig**2.
        elif nu_component == 'dsnb8mev':
            return dsnb8mev_mean_f**2. * (10. ** flux_n - 1.)**2. / dsnb8mev_sig**2.
        elif nu_component == 'reactor':
            return reactor_mean_f**2. * (10. ** flux_n - 1.)**2. / reactor_sig**2.
        elif nu_component == 'geoU':
            return geoU_mean_f**2. * (10. ** flux_n - 1.)**2. / geoU_mean_f**2.
        elif nu_component == 'geoTh':
            return geoT_mean_f**2. * (10. ** flux_n - 1.)**2. / geoT_mean_f**2.
        else:
            return 0.


class Nu_spec(object):
    # Think about defining some of these neutino parameters as variables in constants.py (e.g. mean flux)
    def __init__(self, lab):
        self.lab = lab

    def nu_rate(self, nu_component, er, element_info):

        mT, Z, A, xi = element_info
        conversion_factor = xi / mT * s_to_yr * (0.938 / (1.66 * 10.**-27.)) \
                            * 10**-3. / (0.51 * 10.**14.)**2.
        # Where is this (A-Z)/100 coming from? Should not be there???

        #	print ('nuspec check, component: {}'.format(nu_component))

        diff_rate = np.zeros_like(er)
        for i,e in enumerate(er):
            e_nu_min = np.sqrt(mT * e / 2.)

            if nu_component == 'b8':
                e_nu_max = 16.18 # b8 
                nu_mean_f = 5.58 * 10. ** 6. # b8 cm^-2 s^-1
            elif nu_component == 'b7l1':
                e_nu_max = 0.39
                nu_mean_f = (0.1) * 5.00 * 10. ** 9.
            elif nu_component == 'b7l2':
                e_nu_max = 0.87
                nu_mean_f = (0.9) * 5.00 * 10. ** 9.
            elif nu_component == 'pepl1':
                e_nu_max = 1.45
                nu_mean_f = 1.44 * 10. ** 8.
            elif nu_component == 'hep':
                e_nu_max = 18.77
                nu_mean_f = 8.04 * 10. ** 3.
            elif nu_component == 'pp':
                e_nu_max = 0.42
                nu_mean_f = 5.98 * 10. ** 10.
            elif nu_component == 'o15':
                e_nu_max = 1.73
                nu_mean_f = 2.23 * 10. ** 8.
            elif nu_component == 'n13':
                e_nu_max = 1.20
                nu_mean_f = 2.96 * 10. ** 8.
            elif nu_component == 'f17':
                e_nu_max = 1.74
                nu_mean_f = 5.52 * 10. ** 6.
            elif nu_component == 'atmnue':
                e_nu_max = 9.44 * 10 ** 2
                nu_mean_f = 1.27 * 10. ** 1
            elif nu_component == 'atmnuebar':
                e_nu_max = 9.44 * 10 ** 2
                nu_mean_f = 1.17 * 10. ** 1
            elif nu_component == 'atmnumu':
                e_nu_max = 9.44 * 10 ** 2
                nu_mean_f = 2.46 * 10. ** 1
            elif nu_component == 'atmnumu':
                e_nu_max = 9.44 * 10 ** 2
                nu_mean_f = 2.45 * 10. ** 1
            elif nu_component == 'dsnb3mev':
                e_nu_max = 36.90
                nu_mean_f = 4.55 * 10. ** 1
            elif nu_component == 'dsnb5mev':
                e_nu_max = 57.01
                nu_mean_f = 2.73 * 10. ** 1
            elif nu_component == 'dsnb8mev':
                e_nu_max = 81.91
                nu_mean_f = 1.75 * 10. ** 1
            elif nu_component == 'reactor':
                e_nu_max = 10.
                nu_mean_f = reactor_flux(loc=self.lab)[0]
            elif nu_component == 'geoU':
                e_nu_max = 4.
                nu_mean_f = geo_flux(loc=self.lab, el='U')[0]
            elif nu_component == 'geoTh':
                e_nu_max = 2.252
                nu_mean_f = geo_flux(loc=self.lab, el='Th')[0]

            else:
                return 0.


            diff_rate[i] = romberg(self.nu_recoil_spec, e_nu_min, e_nu_max,
                                   args=(e, mT, Z, A, nu_component))

            #print ('result romb: {}'.format(diff_rate[i]))
            #print ('result quad: {}'.format(quad(self.nu_recoil_spec, e_nu_min, e_nu_max,
            #                   args=(e, mT, Z, A, nu_component),limit=50)[0]))

            diff_rate[i] *= nu_mean_f * conversion_factor

        return diff_rate

    def max_er_from_nu(self, enu, mT):
        # return 2. * enu**2. / (mT + 2. * enu) # -- This formula is in 1307.5458, but it is
        # not consistent with the numbers they use in other papers...
        return 2. * enu**2. / mT

    def nu_recoil_spec(self, enu, er, mT, Z, A, nu_comp):

        if nu_comp == 'b8':
            return self.nu_csec(enu, er, mT, Z, A) * b8nu_spectrum(enu)
        elif nu_comp == 'b7l1':
            return self.nu_csec(enu, er, mT, Z, A) * b7nul1_spectrum(enu)
        elif nu_comp == 'b7l2':
            return self.nu_csec(enu, er, mT, Z, A) * b7nul2_spectrum(enu)
        elif nu_comp == 'pepl1':
            return self.nu_csec(enu, er, mT, Z, A) * pepnul1_spectrum(enu)
        elif nu_comp == 'hep':
            return self.nu_csec(enu, er, mT, Z, A) * hepnu_spectrum(enu)
        elif nu_comp == 'pp':
            return self.nu_csec(enu, er, mT, Z, A) * ppnu_spectrum(enu)
        elif nu_comp == 'o15':
            return self.nu_csec(enu, er, mT, Z, A) * o15nu_spectrum(enu)
        elif nu_comp == 'n13':
            return self.nu_csec(enu, er, mT, Z, A) * n13nu_spectrum(enu)
        elif nu_comp == 'f17':
            return self.nu_csec(enu, er, mT, Z, A) * f17nu_spectrum(enu)
        elif nu_comp == 'atmnue':
            return self.nu_csec(enu, er, mT, Z, A) * atmnue_spectrum(enu)
        elif nu_comp == 'atmnuebar':
            return self.nu_csec(enu, er, mT, Z, A) * atmnuebar_spectrum(enu)
        elif nu_comp == 'atmnumu':
            return self.nu_csec(enu, er, mT, Z, A) * atmnumu_spectrum(enu)
        elif nu_comp == 'atmnumubar':
            return self.nu_csec(enu, er, mT, Z, A) * atmnumubar_spectrum(enu)
        elif nu_comp == 'dsnb3mev':
            return self.nu_csec(enu, er, mT, Z, A) * dsnb3mevnu_spectrum(enu)
        elif nu_comp == 'dsnb5mev':
            return self.nu_csec(enu, er, mT, Z, A) * dsnb5mevnu_spectrum(enu)
        elif nu_comp == 'dsnb8mev':
            return self.nu_csec(enu, er, mT, Z, A) * dsnb8mevnu_spectrum(enu)
        elif nu_comp == 'reactor':
            return self.nu_csec(enu, er, mT, Z, A) * reactor_nu_spectrum(enu)
        elif nu_comp == 'geoU':
            return self.nu_csec(enu, er, mT, Z, A) * geoU_spectrum(enu)
        elif nu_comp == 'geoTh':
            return self.nu_csec(enu, er, mT, Z, A) * geoTh_spectrum(enu)
        else:
            return 0.

    def nu_csec(self, enu, er, mT, Z, A):
        Qw = (A - Z) - (1. - 4. * sw) * Z
        if er < self.max_er_from_nu(enu, mT):
            return gF ** 2. / (4. * np.pi) * Qw**2. * mT * \
                   (1. - mT * er / (2. * enu**2.)) * self.helm_ff(er, A, Z, mT)
        else:
            return 0.

    def helm_ff(self, er, A, Z, mT):
        q = np.sqrt(2. * mT * er) * MeVtofm
        rn = np.sqrt((1.2 * A**(1./3.))**2. - 5.)
        return (3. * np.exp(- q**2. / 2.) * (np.sin(q * rn) - q * rn * np.cos(q * rn)) / (q*rn)**3.)**2.

