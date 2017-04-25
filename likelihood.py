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
            if return_deriv:
                return b8_mean_f**2./b8_sig**2.*(10.**flux_n - 1.)* \
                       np.log(10.)*2.**(flux_n + 1.)*5.**flux_n

            else:
                return b8_mean_f**2. * (10. ** flux_n - 1.)**2. / b8_sig**2.
        elif nu_component == 'b7l1':
            if return_deriv:
                return b7l1_mean_f ** 2. / b7l1_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return b7l1_mean_f**2. * (10. ** flux_n - 1.)**2. / b7l1_sig**2.
        elif nu_component == 'b7l2':
            if return_deriv:
                return b7l2_mean_f ** 2. / b7l2_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return b7l2_mean_f**2. * (10. ** flux_n - 1.)**2. / b7l2_sig**2.
        elif nu_component == 'pepl1':
            if return_deriv:
                return pepl1_mean_f ** 2. / pepl1_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return pepl1_mean_f**2. * (10. ** flux_n - 1.)**2. / pepl1_sig**2.
        elif nu_component == 'hep':
            if return_deriv:
                return hep_mean_f ** 2. / hep_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return hep_mean_f**2. * (10. ** flux_n - 1.)**2. / hep_sig**2.
        elif nu_component == 'pp':
            if return_deriv:
                return pp_mean_f ** 2. / pp_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return pp_mean_f**2. * (10. ** flux_n - 1.)**2. / pp_sig**2.
        elif nu_component == 'o15':
            if return_deriv:
                return o15_mean_f ** 2. / o15_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return o15_mean_f**2. * (10. ** flux_n - 1.)**2. / o15_sig**2.
        elif nu_component == 'n13':
            if return_deriv:
                return n13_mean_f ** 2. / n13_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return n13_mean_f**2. * (10. ** flux_n - 1.)**2. / n13_sig**2.
        elif nu_component == 'f17':
            if return_deriv:
                return f17_mean_f ** 2. / f17_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return f17_mean_f**2. * (10. ** flux_n - 1.)**2. / f17_sig**2.
        elif nu_component == 'atmnue':
            if return_deriv:
                return atmnue_mean_f ** 2. / atmnue_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return atmnue_mean_f**2. * (10. ** flux_n - 1.)**2. / atmnue_sig**2.
        elif nu_component == 'atmnuebar':
            if return_deriv:
                return atmnuebar_mean_f ** 2. / atmnuebar_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return atmnuebar_mean_f**2. * (10. ** flux_n - 1.)**2. / atmnuebar_sig**2.
        elif nu_component == 'atmnumu':
            if return_deriv:
                return atmnumu_mean_f ** 2. / atmnumu_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return atmnumu_mean_f**2. * (10. ** flux_n - 1.)**2. / atmnumu_sig**2.
        elif nu_component == 'atmnumubar':
            if return_deriv:
                return atmnumubar_mean_f ** 2. / atmnumubar_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return atmnumubar_mean_f**2. * (10. ** flux_n - 1.)**2. / atmnumubar_sig**2.
        elif nu_component == 'dsnb3mev':
            if return_deriv:
                return dsnb3mev_mean_f ** 2. / dsnb3mev_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return dsnb3mev_mean_f**2. * (10. ** flux_n - 1.)**2. / dsnb3mev_sig**2.
        elif nu_component == 'dsnb5mev':
            if return_deriv:
                return dsnb5mev_mean_f ** 2. / dsnb5mev_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return dsnb5mev_mean_f**2. * (10. ** flux_n - 1.)**2. / dsnb5mev_sig**2.
        elif nu_component == 'dsnb8mev':
            if return_deriv:
                return dsnb8mev_mean_f ** 2. / dsnb8mev_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return dsnb8mev_mean_f**2. * (10. ** flux_n - 1.)**2. / dsnb8mev_sig**2.
        elif nu_component == 'reactor':
            if return_deriv:
                return reactor_mean_f ** 2. / reactor_sig ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return reactor_mean_f**2. * (10. ** flux_n - 1.)**2. / reactor_sig**2.
        elif nu_component == 'geoU':
            if return_deriv:
                return geoU_mean_f ** 2. / geoU_mean_f ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return geoU_mean_f**2. * (10. ** flux_n - 1.)**2. / geoU_mean_f**2.
        elif nu_component == 'geoTh':
            if return_deriv:
                return geoT_mean_f ** 2. / geoT_mean_f ** 2. * (10. ** flux_n - 1.) * \
                       np.log(10.) * 2.** (flux_n + 1.) * 5. ** flux_n
            else:
                return geoT_mean_f**2. * (10. ** flux_n - 1.)**2. / geoT_mean_f**2.
        else:
            return 0.


class Nu_spec(object):
    # Think about defining some of these neutino parameters as variables in constants.py (e.g. mean flux)
    def __init__(self, lab):
        self.lab = lab
        self.nu_lines = ['b7l1', 'b7l2','pepl1']
        self.line = [ 0.380, 0.860, 1.440]

    def nu_rate(self, nu_component, er, element_info):

        mT, Z, A, xi = element_info
        conversion_factor = xi / mT * s_to_yr * (0.938 / (1.66 * 10.**-27.)) \
                            * 10**-3. / (0.51 * 10.**14.)**2.

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
                e_nu_max = 3.99
                nu_mean_f = geo_flux(loc=self.lab, el='U')[0]
            elif nu_component == 'geoTh':
                e_nu_max = 2.26
                nu_mean_f = geo_flux(loc=self.lab, el='Th')[0]

            else:
                return 0.

            if nu_component not in self.nu_lines:
                ergs = np.logspace(np.log10(e_nu_min), np.log10(e_nu_max), 100)
                diff_rate[i] = np.trapz(self.nu_recoil_spec(ergs, e, mT, Z, A, nu_component), ergs)
                #diff_rate[i] = romberg(self.nu_recoil_spec, e_nu_min, e_nu_max, args=(e, mT, Z, A, nu_component))

            else:
                if nu_component == self.nu_lines[0]:
                    diff_rate[i] = self.nu_recoil_spec(self.line[0], e, mT, Z, A, nu_component)
                elif nu_component == self.nu_lines[1]:
                    diff_rate[i] = self.nu_recoil_spec(self.line[1], e, mT, Z, A, nu_component)
                elif nu_component == self.nu_lines[2]:
                    diff_rate[i] = self.nu_recoil_spec(self.line[2], e, mT, Z, A, nu_component)

            diff_rate[i] *= nu_mean_f * conversion_factor

        return diff_rate

    def max_er_from_nu(self, enu, mT):
        return 2. * enu**2. / (mT + 2. * enu * 1e-3)

    def nu_recoil_spec(self, enu, er, mT, Z, A, nu_comp):

        if nu_comp == 'b8':
            return self.nu_csec(enu, er, mT, Z, A) * b8nu_spectrum(enu)
        elif nu_comp == 'b7l1':
            if enu == self.line[0]:
                return self.nu_csec(enu, er, mT, Z, A)
        elif nu_comp == 'b7l2':
            if enu == self.line[1]:
                return self.nu_csec(enu, er, mT, Z, A)
        elif nu_comp == 'pepl1':
            if enu == self.line[2]:
                return self.nu_csec(enu, er, mT, Z, A)
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

