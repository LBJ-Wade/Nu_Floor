"""
Runs nuetrino floor analysis

Model options: [sigma_si, sigma_sd, sigma_anapole, sigma_magdip, sigma_elecdip,
                sigma_LS, sigma_f1, sigma_f2, sigma_f3, sigma_si_massless,
                sigma_sd_massless, sigma_anapole_massless, sigma_magdip_massless,
                sigma_elecdip_massless, sigma_LS_massless, sigma_f1_massless,
                sigma_f2_massless, sigma_f3_massless]

Element options:  ['germanium', 'xenon', 'argon', 'silicon', 'fluorine']

Exposure in Ton-year

Delta = 0. -- Note: This code doesn't yet work for inelastic kinematics

"""

import numpy as np
from experiments import *
import numpy.random as random
from likelihood import *
from scipy.optimize import minimize, basinhopping
from scipy.stats import chi2
from scipy.integrate import quad
from scipy.interpolate import interp1d, RectBivariateSpline
from rate_UV import *
from helpers import *
import os
from scipy.stats import poisson

path = os.getcwd()

QUIET = False


def nu_floor(sig_low, sig_high, n_sigs=10, model="sigma_si", mass=6., fnfp=1.,
             element='germanium', exposure=1., delta=0., GF=False, time_info=False,
             file_tag='', n_runs=20):

    sig_list = np.logspace(np.log10(sig_low), np.log10(sig_high), n_sigs)

    for sigmap in sig_list:

        coupling = "fnfp" + model[5:]

        print 'Run Info:'
        print 'Experiment: ', element
        print 'Model: ', model
        print 'Coupling: ', coupling, fnfp
        print 'Mass: {:.0f}, Sigma: {:.2e}'.format(mass, sigmap)

        file_info = path + '/Saved_Files/'
        file_info += element + '_' + model + '_' + coupling + '_{:.0f}'.format(fnfp)
        file_info += '_Exposure_{:.1f}_tonyr_DM_Mass_{:.0f}_GeV'.format(exposure, mass)
        file_info += file_tag + '.dat'
        print 'Output File: ', file_info
        print '\n'
        experiment_info, Qmin, Qmax = Element_Info(element)

        drdq_params = default_rate_parameters.copy()
        drdq_params['element'] = element
        drdq_params['mass'] = mass
        drdq_params[model] = sigmap
        drdq_params[coupling] = fnfp
        drdq_params['delta'] = delta
        drdq_params['GF'] = GF
        drdq_params['time_info'] = time_info

        # 3\sigma for Chi-square Dist with 1 DoF means q = 9.0
        q_goal = 9.0

        # make sure there are enough points for numerical accuracy/stability
        er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), 500)
        time_list = np.zeros_like(er_list)

        dm_spec = dRdQ(er_list, time_list, **drdq_params) * 10. ** 3. * s_to_yr
        dm_rate = R(Qmin=Qmin, Qmax=Qmax, **drdq_params) * 10. ** 3. * s_to_yr * exposure
        dm_pdf = dm_spec / dm_rate
        cdf_dm = dm_pdf.cumsum()
        cdf_dm /= cdf_dm.max()
        dm_events_sim = int(dm_rate * exposure)


        # TODO generalize beyond B8
        nu_comp = 'B8'
        # neutrino ER spectrum
        nuspec = np.zeros_like(er_list)
        for iso in experiment_info:
            nuspec += Nu_spec().nu_rate(nu_comp, er_list, iso)

        nu_rate = np.trapz(nuspec, er_list)
        nu_pdf = nuspec / nu_rate
        cdf_nu = nu_pdf.cumsum()
        cdf_nu /= cdf_nu.max()
        Nu_events_sim = int(nu_rate * exposure)



        tstat_arr = np.zeros(n_runs)
        # While loop goes here. Fill tstat_arr for new sims and extract median/mean
        nn = 0
        while nn < n_runs:
            print 'Run {:.0f} of {:.0f}'.format(nn, n_runs)
            nevts_dm = poisson.rvs(int(dm_events_sim))
            nevts_n = poisson.rvs(int(Nu_events_sim))
            if not QUIET:
                print 'Predicted Number of Nu events: {}'.format(Nu_events_sim)
                print 'Predicted Number of DM events: {}'.format(dm_events_sim)

            # Simulate events
            Nevents = nevts_n + nevts_dm
            if not QUIET:
                print 'Simulation {:.0f} events...'.format(Nevents)
            u = random.rand(Nevents)
            # Generalize to rejection sampling algo for time implimentation
            e_sim = np.zeros(Nevents)
            for i in range(Nevents):
                if i < int(nevts_n):
                    e_sim[i] = er_list[np.absolute(cdf_nu - u[i]).argmin()]
                else:
                    e_sim[i] = er_list[np.absolute(cdf_dm - u[i]).argmin()]
            times = np.zeros_like(e_sim)
            #print e_sim

            if not QUIET:
                print 'Running Likelihood Analysis...'
            # Minimize likelihood -- MAKE SURE THIS MINIMIZATION DOESNT FAIL. CONSIDER USING GRADIENT INFO
            like_init_nodm = Likelihood_analysis(model, coupling, mass, 0., fnfp,
                                                 exposure, element, experiment_info, e_sim, times,
                                                 Qmin=Qmin, Qmax=Qmax, time_info=time_info, GF=False)
            max_nodm = minimize(like_init_nodm.likelihood, np.array([0.]), args=(np.array([-100.])), tol=0.001)
            #print max_nodm

            like_init_dm = Likelihood_analysis(model, coupling, mass, 1., fnfp,
                                               exposure, element, experiment_info, e_sim, times,
                                               Qmin=Qmin, Qmax=Qmax, time_info=time_info, GF=False)
            max_dm = minimize(like_init_dm.like_multi_wrapper, np.array([0., np.log10(sigmap)]), tol=0.001,
                              jac=False)

            if not QUIET:
                print 'BF Neutrino normalization without DM: {:.2e}'.format(10.**max_nodm.x[0])
                print 'BF Neutrino normalization with DM: {:.2e}'.format(10.**max_dm.x[0])
                print 'BF DM sigma_p: {:.2e} \n\n'.format(10.**max_dm.x[1])

            test_stat = np.max([max_nodm.fun - max_dm.fun, 0.])

            pval = chi2.sf(test_stat,1)

            if not QUIET:
                print 'TS: ', test_stat
                print 'p-value: ', pval

            tstat_arr[nn] = test_stat
            nn += 1

        print 'FINISHED CYCLE \n'
        print 'True DM mass: ', mass
        print 'True DM sigma_p: ', sigmap
        print 'Median Q: {:.2f}'.format(np.median(tstat_arr))
        print 'Mean Q: {:.2f}'.format(np.mean(tstat_arr))

        if os.path.exists(file_info):
            load_old = np.loadtxt(file_info)
            new_arr = np.vstack((load_old, np.array([np.log10(sigmap), np.median(tstat_arr)])))
            new_arr = new_arr[new_arr[:, 0].argsort()]
            np.savetxt(file_info, new_arr)
        else:
            np.savetxt(file_info, np.array([np.log10(sigmap), np.median(tstat_arr)]))

    return

