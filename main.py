"""
Runs nuetrino floor analysis

Model options: [sigma_si, sigma_sd, sigma_anapole, sigma_magdip, sigma_elecdip,
				sigma_LS, sigma_f1, sigma_f2, sigma_f3, sigma_si_massless,
				sigma_sd_massless, sigma_anapole_massless, sigma_magdip_massless,
				sigma_elecdip_massless, sigma_LS_massless, sigma_f1_massless,
				sigma_f2_massless, sigma_f3_massless]

Element options:  ['germanium', 'xenon', 'argon', 'sodium', 'fluorine', 'iodine', 'neon']

Exposure in Ton-year


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
import time

path = os.getcwd()

QUIET = False

xenLAB = 'LZ'

def nu_floor(sig_low, sig_high, n_sigs=10, model="sigma_si", mass=6., fnfp=1.,
            element='germanium', exposure=1., delta=0., GF=False, time_info=False,
            file_tag='', n_runs=20):

    #start_time = time.time()
    sig_list = np.logspace(np.log10(sig_low), np.log10(sig_high), n_sigs)

    testq = 0

    print 'Run Info:'
    print 'Experiment: ', element
    print 'Model: ', model
    coupling = "fnfp" + model[5:]
    print 'Coupling: ', coupling, fnfp
    print 'Mass: {:.0f}'.format(mass)
    print '\n'

    file_info = path + '/Saved_Files/'
    file_info += element + '_' + model + '_' + coupling + '_{:.2f}'.format(fnfp)
    file_info += '_Exposure_{:.2f}_tonyr_DM_Mass_{:.2f}_GeV'.format(exposure, mass)
    file_info += file_tag + '.dat'
    print 'Output File: ', file_info
    print '\n'
    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenLAB)

    mindm = np.zeros(len(experiment_info[:, 0]))
    for i, iso in enumerate(experiment_info):
        mindm[i] = MinDMMass(iso[0], delta, Qmin, vesc=533.+232.)
    MinDM = np.min(mindm)
    print 'Minimum DM mass: ', MinDM
    if mass < MinDM:
        print 'Mass too small...'
        exit()
    # 3\sigma for Chi-square Dist with 1 DoF means q = 9.0
    q_goal = 9.0

    # make sure there are enough points for numerical accuracy/stability
    er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), 100)
    time_list = np.zeros_like(er_list)

    nu_comp = ['b8', 'b7l1', 'b7l2', 'pepl1', 'hep', 'pp', 'o15', 'n13', 'f17', 'atmnue',
               'atmnuebar', 'atmnumu', 'atmnumubar', 'dsnb3mev', 'dsnb5mev', 'dsnb8mev',
               'reactor', 'geoU', 'geoTh']

    nu_lines = ['b7l1', 'b7l2', 'pepl1']
    line_flux = [(0.1) * 5.00 * 10. ** 9., (0.9) * 5.00 * 10. ** 9., 1.44 * 10. ** 8.]
    e_lines = [0.380, 0.860, 1.440]

    nu_contrib = len(nu_comp)

    nuspec = np.zeros(nu_contrib, dtype=object)
    nu_rate = np.zeros(nu_contrib, dtype=object)
    nu_pdf = np.zeros(nu_contrib, dtype=object)
    cdf_nu = np.zeros(nu_contrib, dtype=object)
    Nu_events_sim = np.zeros(nu_contrib)

    for i in range(nu_contrib):
        nuspec[i] = np.zeros_like(er_list)

    for iso in experiment_info:
        for i in range(nu_contrib):
            nuspec[i] += Nu_spec(labor).nu_rate(nu_comp[i], er_list, iso)


    for i in range(nu_contrib):
        nu_rate[i] = np.trapz(nuspec[i], er_list)


        print nu_comp[i], nu_rate[i]
        if nu_rate[i] > 0.:
            nu_pdf[i] = nuspec[i] / nu_rate[i]
            cdf_nu[i] = nu_pdf[i].cumsum()
            cdf_nu[i] /= cdf_nu[i].max()
            Nu_events_sim[i] = int(nu_rate[i] * exposure)

    nevts_n = np.zeros(nu_contrib)
    print '\n \n'

    for sigmap in sig_list:

        print 'Sigma: {:.2e}'.format(sigmap)

        drdq_params = default_rate_parameters.copy()
        drdq_params['element'] = element
        drdq_params['mass'] = mass
        drdq_params[model] = sigmap
        drdq_params[coupling] = fnfp
        drdq_params['delta'] = delta
        drdq_params['GF'] = GF
        drdq_params['time_info'] = time_info
        #
        #
        dm_spec = dRdQ(er_list, time_list, **drdq_params) * 10. ** 3. * s_to_yr
        dm_rate = R(Qmin=Qmin, Qmax=Qmax, **drdq_params) * 10. ** 3. * s_to_yr * exposure
        dm_pdf = dm_spec / dm_rate
        cdf_dm = dm_pdf.cumsum()
        cdf_dm /= cdf_dm.max()
        dm_events_sim = 0.
        dm_events_sim = int(dm_rate * exposure)

        if dm_events_sim < 1.:
            continue

        tstat_arr = np.zeros(n_runs)
        nn = 0

        fails = np.array([])
        while nn < n_runs:
            for i in range(nu_contrib):
                try:
                    nevts_n[i] = poisson.rvs(int(Nu_events_sim[i]))
                except ValueError:
                    nevts_n[i] = 0

            Nevents = int(sum(nevts_n))

            u = random.rand(Nevents)
            e_sim = np.zeros(Nevents)
            sum_nu_evts = np.zeros(nu_contrib)

            for i in range(nu_contrib):
                sum_nu_evts[i] = np.sum(nevts_n[:i + 1])
            sum_nu_evts = np.insert(sum_nu_evts, 0, -1)
            j = 0

            for i in range(Nevents):
                if i < sum(nevts_n):
                    for j in range(nu_contrib + 1):
                        if sum_nu_evts[j] <= i < sum_nu_evts[j + 1]:
                            if nu_comp[j] not in nu_lines:
                                e_sim[i] = er_list[np.absolute(cdf_nu[j] - u[i]).argmin()]
                            else:
                                if nu_comp[j] == nu_lines[0]:
                                    e_sim[i] = e_lines[0]
                                elif nu_comp[j] == nu_lines[1]:
                                    e_sim[i] = e_lines[1]
                                elif nu_comp[j] == nu_lines[2]:
                                    e_sim[i] = e_lines[2]

            print 'Run {:.0f} of {:.0f}'.format(nn + 1, n_runs)
            try:
                nevts_dm = poisson.rvs(int(dm_events_sim))
            except ValueError:
                nevts_dm = 0

            if not QUIET:
                print 'Predicted Number of Nu events: {}'.format(sum(Nu_events_sim))
                print 'Predicted Number of DM events: {}'.format(dm_events_sim)

            # Simulate events
            print('Evaluated Events: Neutrino {:.0f}, DM {:.0f}'.format(sum(nevts_n), nevts_dm))

            u = random.rand(nevts_dm)
            # Generalize to rejection sampling algo for time implimentation


            for i in range(nevts_dm):
                 e_sim = np.append(e_sim, er_list[np.absolute(cdf_dm - u[i]).argmin()])

            times = np.zeros_like(e_sim)
            #print e_sim

            if not QUIET:
                print 'Running Likelihood Analysis...'
            # Minimize likelihood -- MAKE SURE THIS MINIMIZATION DOESNT FAIL. CONSIDER USING GRADIENT INFO
            nu_bnds = [(-3.0, 3.0)] * nu_contrib
            dm_bnds = nu_bnds + [(-60., -30.)]
            like_init_nodm = Likelihood_analysis(model, coupling, mass, 0., fnfp,
                                                 exposure, element, experiment_info,
                                                 e_sim, times, nu_comp, labor,
                                                 nu_contrib,
                                                 Qmin, Qmax, time_info=time_info, GF=False)

            max_nodm = minimize(like_init_nodm.likelihood, np.zeros(nu_contrib),
                                args=(np.array([-100.])), tol=1e-5, method='SLSQP',
                                options={'maxiter': 100}, bounds=nu_bnds,
                                jac=like_init_nodm.like_gradi)

            like_init_dm = Likelihood_analysis(model, coupling, mass, 1., fnfp,
                                               exposure, element, experiment_info, e_sim, times, nu_comp, labor,
                                               nu_contrib,
                                               Qmin, Qmax, time_info=time_info, GF=False)

            max_dm = minimize(like_init_dm.like_multi_wrapper,
                              np.concatenate((np.zeros(nu_contrib),np.array([np.log10(sigmap)]))),
                              tol=1e-5, method='SLSQP', bounds=dm_bnds,
                              options={'maxiter': 100}, jac=like_init_dm.likegrad_multi_wrapper)
            print 'Minimizaiton Success: ', max_nodm.success, max_dm.success
            print 'DM Vals: ', max_dm.x
            if not max_nodm.success or not max_dm.success:
                fails = np.append(fails, nn)

            test_stat = np.max([max_nodm.fun - max_dm.fun, 0.])

            pval = chi2.sf(test_stat, 1)

            if not QUIET:
                print 'TS: ', test_stat
                print 'p-value: ', pval
                print '\n \n'
            tstat_arr[nn] = test_stat
            nn += 1

        mask = np.array([(i in fails) for i in xrange(len(tstat_arr))])
        tstat_arr = tstat_arr[~mask]
        #print 'Tstat Array'
        print tstat_arr

        print 'FINISHED CYCLE \n'
        print 'True DM mass: ', mass
        print 'True DM sigma_p: ', sigmap
        #print("--- %s seconds ---" % (time.time() - start_time))


        if len(tstat_arr) > 0:
            print 'Median Q: {:.2f}'.format(np.median(tstat_arr))
            print 'Mean Q: {:.2f}\n'.format(np.mean(tstat_arr))
            print 'T-stat Array:', tstat_arr

            testq = float(np.sum(tstat_arr > q_goal)) / float(len(tstat_arr))

            print 'testq (mean, end n cycle): {}'.format(testq)

            if testq > 0.99:
                print 'testq: {} --> BREAK'.format(testq)
                print '~~~~~~~~~~~~~~~~~~~~~MOVING ON~~~~~~~~~~~~~~~~~~~~~'
                print '\n'
                if os.path.exists(file_info):
                    load_old = np.loadtxt(file_info)
                    new_arr = np.vstack((load_old, np.array([np.log10(sigmap), testq])))
                    new_arr = new_arr[new_arr[:, 0].argsort()]
                    np.savetxt(file_info, new_arr)
                else:
                    np.savetxt(file_info, np.array([np.log10(sigmap), testq]))
                break

            else:
                print 'testq: {} --> WRITE'.format(testq)
                print '~~~~~~~~~~~~~~~~~~~~~MOVING ON~~~~~~~~~~~~~~~~~~~~~'
                print '\n'
                if os.path.exists(file_info):
                    load_old = np.loadtxt(file_info)
                    new_arr = np.vstack((load_old, np.array([np.log10(sigmap), testq])))
                    new_arr = new_arr[new_arr[:, 0].argsort()]
                    np.savetxt(file_info, new_arr)
                else:
                    np.savetxt(file_info, np.array([np.log10(sigmap), testq]))
        else:
            print 'T-stat Array does not have any non-zero values...\n \n'

    return
