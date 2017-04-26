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
from constants import *

path = os.getcwd()


gF = 1.16637 * 10. ** -5. # Gfermi in GeV^-2
sw = 0.2312 # sin(theat_weak)^2
MeVtofm = 0.0050677312
s_to_yr = 3.154*10.**7.


class Likelihood_analysis(object):

    def __init__(self, model, coupling, mass, dm_sigp, fnfp, exposure, element, isotopes,
                 energies, times, nu_names, lab, nu_spec, Qmin, Qmax, time_info=False, GF=False):

        self.nu_lines = ['b7l1', 'b7l2', 'pepl1']
        self.line = [0.380, 0.860, 1.440]


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


        self.dm_recoils = dRdQ(energies, times, **like_params) * 10.**3. * s_to_yr

        self.dm_integ = R(Qmin=self.Qmin, Qmax=self.Qmax, **like_params) * 10.**3. * s_to_yr

        eng_lge = np.logspace(np.log10(self.Qmin), np.log10(self.Qmax), 200)

        for i in range(nu_spec):
            nu_resp[i] = np.zeros_like(eng_lge)

        for iso in isotopes:
            for j in range(nu_spec):
                nu_resp[j] += Nu_spec(self.lab).nu_rate(nu_names[j], eng_lge, iso)

        for i in range(nu_spec):
            #linear interpolation for the 3 lines
            self.nu_resp[i] = interp1d(eng_lge, np.abs(nu_resp[i]), kind='cubic', bounds_error=False, fill_value=0.)
            self.nu_int_resp[i] = np.trapz(self.nu_resp[i](eng_lge), eng_lge)


    def like_multi_wrapper(self, norms, grad=False):
        nu_norm = np.zeros(self.nu_spec, dtype=object)
        for i in range(self.nu_spec):
            nu_norm[i] = norms[i]
        sig_dm = norms[-1]
        return self.likelihood(nu_norm, sig_dm, return_grad=False)

    def test_num_events(self, nu_norm, sig_dm):
        print 'DM events Predicted: ', 10. ** sig_dm * self.dm_integ * self.exposure
        for i in range(self.nu_spec):
            nu_events = 10. ** nu_norm[i] * self.exposure * self.nu_int_resp[i]
            print 'Events from ' + self.nu_names[i] + ': ', nu_events
        return

    def likelihood(self, nu_norm, sig_dm, return_grad=False):
        # - 2 log likelihood
        # nu_norm in units of cm^-2 s^-1, sig_dm in units of cm^2
        like = 0.
        diff_nu = np.zeros(self.nu_spec, dtype=object)
        nu_events = np.zeros(self.nu_spec, dtype=object)

        n_obs = len(self.events)
        # total rate contribution

        dm_events = 10. ** sig_dm * self.dm_integ * self.exposure
        for i in range(self.nu_spec):
            nu_events[i] = 10. ** nu_norm[i] * self.exposure * self.nu_int_resp[i]

        like += 2. * (dm_events + sum(nu_events))

        # nu normalization contribution
        for i in range(self.nu_spec):
            like += self.nu_gaussian(self.nu_names[i], nu_norm[i])

        if self.element == 'fluorine':
            return like

        # Differential contribution
        diff_dm = self.dm_recoils * self.exposure

        for i in range(self.nu_spec):
            diff_nu[i] = self.nu_resp[i](self.events) * self.exposure

        lg_vle = (10. ** sig_dm * diff_dm + np.dot(list(map(lambda x:10**x,nu_norm)),diff_nu)) #nu norm array

        for i in range(len(lg_vle)):
            if lg_vle[i] > 0.:
                like += -2. * np.log(lg_vle[i])

        return like

    def likegrad_multi_wrapper(self, norms):
        nu_norm = np.zeros(self.nu_spec, dtype=object)
        for i in range(self.nu_spec):
            nu_norm[i] = norms[i]
        sig_dm = norms[-1]
        return self.like_gradi(nu_norm, sig_dm, ret_just_nu=False)

    def like_gradi(self, nu_norm, sig_dm, ret_just_nu=True, ret_just_dm=False):
        grad_x = 0.
        diff_nu = np.zeros(self.nu_spec, dtype=object)
        grad_nu = np.zeros(self.nu_spec)
        nu_events = np.zeros(self.nu_spec, dtype=object)

        n_obs = len(self.events)

        dm_events = 10. ** sig_dm * self.dm_integ * self.exposure

        grad_x += 2. * np.log(10.) * dm_events
        for i in range(self.nu_spec):
            grad_nu[i] += 2. * np.log(10.) * 10. ** nu_norm[i] * self.exposure * self.nu_int_resp[i]

        for i in range(self.nu_spec):
            grad_nu[i] += self.nu_gaussian(self.nu_names[i], nu_norm[i], return_deriv=True)

        if self.element != 'fluorine':
            diff_dm = self.dm_recoils * self.exposure
            for i in range(self.nu_spec):
                diff_nu[i] = self.nu_resp[i](self.events) * self.exposure

            lg_vle = (10. ** sig_dm * diff_dm + np.dot(list(map(lambda x: 10 ** x, nu_norm)), diff_nu))

            for i in range(len(lg_vle)):
                grad_x += -2. * np.log(10.) * diff_dm[i] * 10. ** sig_dm / lg_vle[i]
            for i in range(self.nu_spec):
                for j in range(len(lg_vle)):
                    grad_nu[i] += -2. * np.log(10.) * diff_nu[i][j] * 10**nu_norm[i] / lg_vle[j]


        if ret_just_nu:
            return grad_nu
        else:
            if ret_just_dm:
                return np.array([grad_x])
            else:
                return np.concatenate((grad_nu, np.array([grad_x])))


    def nu_gaussian(self, nu_component, flux_n, return_deriv=False):
        # - 2 log of gaussian flux norm comp

        if nu_component == "reactor":
            nu_mean_f, nu_sig = reactor_flux(loc=self.lab)
        elif nu_component == "geoU":
            nu_mean_f, nu_sig = geo_flux(loc=self.lab, el='U')
        elif nu_component == "geoTh":
            nu_mean_f, nu_sig = geo_flux(loc=self.lab, el='Th')
        else:
            nu_mean_f = NEUTRINO_MEANF[nu_component]
            nu_sig = NEUTRINO_SIG[nu_component]

        if return_deriv:
            return nu_mean_f**2./nu_sig**2.*(10.**flux_n - 1.)* \
                       np.log(10.)*2.**(flux_n + 1.)*5.**flux_n
        else:
            return nu_mean_f**2. * (10. ** flux_n - 1.)**2. / nu_sig**2.



class Nu_spec(object):
    # Think about defining some of these neutino parameters as variables in constants.py (e.g. mean flux)
    def __init__(self, lab):
        self.lab = lab
        self.nu_lines = ['b7l1', 'b7l2','pepl1']
        self.line = [0.380, 0.860, 1.440]


    def nu_spectrum_enu(self, nu_component, emin=0.1):
        if 'atm' in nu_component:
            e_nu_min = 13. + 1e-3
        elif nu_component == 'reactor':
            e_nu_min = 0.5 + 1e-3
        else:
            e_nu_min = emin

        e_nu_max = NEUTRINO_EMAX[nu_component] + 10.

        e_list = np.logspace(np.log10(e_nu_min), np.log10(e_nu_max), 500)
        if nu_component == "reactor":
            nu_mean_f = reactor_flux(loc=self.lab)[0]
        elif nu_component == "geoU":
            nu_mean_f = geo_flux(loc=self.lab, el='U')[0]
        elif nu_component == "geoTh":
            nu_mean_f = geo_flux(loc=self.lab, el='Th')[0]
        else:
            nu_mean_f = NEUTRINO_MEANF[nu_component]

        if nu_component not in self.nu_lines:
            return e_list, NEUTRINO_SPEC[nu_component](e_list) * nu_mean_f
        else:
            for i in range(len(self.line)):
                if nu_component == self.nu_lines[i]:
                    return np.array([self.line[i],self.line[i]+1e-5]), np.array([nu_mean_f,nu_mean_f])


    def nu_rate(self, nu_component, er, element_info):

        mT, Z, A, xi = element_info
        conversion_factor = xi / mT * s_to_yr * (0.938 / (1.66 * 10.**-27.)) \
                            * 10**-3. / (0.51 * 10.**14.)**2.


        diff_rate = np.zeros_like(er)
        for i,e in enumerate(er):
            if 'atm' in nu_component:
                e_nu_min = 13.
            elif nu_component == 'reactor':
                e_nu_min = 0.5
            else:
                e_nu_min = np.sqrt(mT * e / 2.)

            e_nu_max = NEUTRINO_EMAX[nu_component]
            if nu_component == "reactor":
                nu_mean_f = reactor_flux(loc=self.lab)[0]
            elif nu_component == "geoU":
                nu_mean_f = geo_flux(loc=self.lab, el='U')[0]
            elif nu_component == "geoTh":
                nu_mean_f = geo_flux(loc=self.lab, el='Th')[0]
            else:
                nu_mean_f = NEUTRINO_MEANF[nu_component]

            if nu_component not in self.nu_lines:
                ergs = np.logspace(np.log10(e_nu_min), np.log10(e_nu_max), 100)
                diff_rate[i] = np.trapz(self.nu_recoil_spec(ergs, e, mT, Z, A, nu_component), ergs)
                #diff_rate[i] = romberg(self.nu_recoil_spec, e_nu_min, e_nu_max, args=(e, mT, Z, A, nu_component))

            else:
                for j in range(len(self.line)):
                    if nu_component == self.nu_lines[j]:
                        diff_rate[i] = self.nu_recoil_spec(self.line[j], e, mT, Z, A, nu_component)

            diff_rate[i] *= nu_mean_f * conversion_factor

        return diff_rate

    def max_er_from_nu(self, enu, mT):
        return 2. * enu**2. / (mT + 2. * enu * 1e-3)

    def nu_recoil_spec(self, enu, er, mT, Z, A, nu_comp):
        if nu_comp in self.nu_lines:
            for i in range(len(self.line)):
                if enu == self.line[i] and nu_comp == self.nu_lines[i]:
                    return self.nu_csec(enu, er, mT, Z, A)
        else:
            return self.nu_csec(enu, er, mT, Z, A) * NEUTRINO_SPEC[nu_comp](enu)


    def nu_csec(self, enu, er, mT, Z, A):
        # enu can be array, er cannot be
        Qw = (A - Z) - (1. - 4. * sw) * Z
        if type(enu) is not np.ndarray:
            enu = np.array([enu])
        ret = np.zeros_like(enu)
        for i,en in enumerate(enu):
            if er < self.max_er_from_nu(en, mT):
                ret[i] = gF ** 2. / (4. * np.pi) * Qw**2. * mT * \
                        (1. - mT * er / (2. * en**2.)) * self.helm_ff(er, A, Z, mT)
        return ret

    def helm_ff(self, er, A, Z, mT):
        q = np.sqrt(2. * mT * er) * MeVtofm
        rn = np.sqrt((1.2 * A**(1./3.))**2. - 5.)
        return (3. * np.exp(- q**2. / 2.) * (np.sin(q * rn) - q * rn * np.cos(q * rn)) / (q*rn)**3.)**2.

