"""
Runs nuetrino floor analysis

Model options: [sigma_si, sigma_sd, sigma_anapole, sigma_magdip, sigma_elecdip,
				sigma_LS, sigma_f1, sigma_f2, sigma_f3, sigma_si_massless,
				sigma_sd_massless, sigma_anapole_massless, sigma_magdip_massless,
				sigma_elecdip_massless, sigma_LS_massless, sigma_f1_massless,
				sigma_f2_massless, sigma_f3_massless]

Element options:  ['Germanium', 'Xenon', 'Argon', 'Sodium', 'Fluorine', 'Iodine', 'Neon']

Exposure in Ton-year


"""

import numpy as np
from experiments import *
import numpy.random as random
from likelihood import *
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.integrate import quad
from scipy.interpolate import interp1d
from rate_UV import *
from helpers import *
import os
from scipy.stats import poisson
import time
from test_plots import *
from constants import *
from scipy.stats import gaussian_kde

path = os.getcwd()
QUIET = False
xenLAB = 'LZ'


def nu_floor(sig_low, sig_high, n_sigs=10, model="sigma_si", mass=6., fnfp=1.,
             element='Germanium', exposure=1., delta=0., GF=False, time_info=False,
             file_tag='', n_runs=20, Eth=''):

    #start_time = time.time()
    testq = 0

    print 'Run Info:'
    print 'Experiment: ', element
    print 'Model: ', model
    coupling = "fnfp" + model[5:]
    print 'Coupling: ', coupling, fnfp
    print 'Mass: {:.0f}'.format(mass)
    print '\n'

    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenLAB)
    if Eth > 0:
        Qmin = Eth

    file_info = path + '/Saved_Files/'
    file_info += element + '_' + model + '_' + coupling + '_{:.2f}'.format(fnfp)
    file_info += '_Exposure_{:.2f}_tonyr_DM_Mass_{:.2f}_GeV'.format(exposure, mass)
    file_info += '_Eth_{:.2f}_'.format(Qmin) + labor + '_' + file_tag + '.dat'
    print 'Output File: ', file_info
    print '\n'


    mindm = np.zeros(len(experiment_info[:, 0]))
    for i, iso in enumerate(experiment_info):
        mindm[i] = MinDMMass(iso[0], delta, Qmin, vesc=533.+232.)
    MinDM = np.min(mindm)
    print 'Minimum DM mass: ', MinDM
    if mass < MinDM + 0.5:
        print 'Mass too small...'
        exit()
    # 3\sigma for Chi-square Dist with 1 DoF means q = 9.0
    q_goal = 9.0

    # make sure there are enough points for numerical accuracy/stability
    er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), 300)
    time_list = np.zeros_like(er_list)

    nu_comp = ['b8', 'b7l1', 'b7l2', 'pepl1', 'hep', 'pp', 'o15', 'n13', 'f17', 'atm',
               'dsnb3mev', 'dsnb5mev', 'dsnb8mev', 'reactor', 'geoU', 'geoTh']
    keep_nus = []
    for i in range(len(nu_comp)):
        if Nu_spec(Nu_spec).max_er_from_nu(NEUTRINO_EMAX[nu_comp[i]], experiment_info[0][0]) > Qmin:
            keep_nus.append(i)
    nu_comp = [x for i,x in enumerate(nu_comp) if i in keep_nus]
    nu_contrib = len(nu_comp)
    print 'Neutrinos Considered: ', nu_comp

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
            cdf_nu[i] = np.zeros_like(nu_pdf[i])
            for j in range(len(nu_pdf[i])):
                cdf_nu[i][j] = np.trapz(nu_pdf[i][:j],er_list[:j])

            cdf_nu[i] /= cdf_nu[i].max()
            Nu_events_sim[i] = int(nu_rate[i] * exposure)

    nevts_n = np.zeros(nu_contrib)
    print '\n \n'

    win = False
    sig_list = []
    while not win:
        print 'Sigma List: ', sig_list
        sig, win = adaptive_samples(np.log10(sig_low), np.log10(sig_high), sig_list)
        sigmap = 10.**sig
        if len(sig_list) > n_sigs:
            win = True
            break

        print 'Sigma: {:.2e}'.format(sigmap)

        try:
            check = np.loadtxt(file_info)
            try:
                if np.log10(sigmap) in check[:, 0]:
                    fill = check[np.log10(sigmap) == check[:, 0]]
                    sig_list.append(fill)
                    continue
            except IndexError:
                if np.log10(sigmap) == check[0]:
                    sig_list.append(check)
                    continue

        except IOError:
            pass

        drdq_params = default_rate_parameters.copy()
        drdq_params['element'] = element
        drdq_params['mass'] = mass
        drdq_params[model] = sigmap
        drdq_params[coupling] = fnfp
        drdq_params['delta'] = delta
        drdq_params['GF'] = GF
        drdq_params['time_info'] = time_info

        dm_spec = dRdQ(er_list, time_list, **drdq_params) * 10. ** 3. * s_to_yr
        dm_rate = R(Qmin=Qmin, Qmax=Qmax, **drdq_params) * 10. ** 3. * s_to_yr
        dm_pdf = dm_spec / dm_rate

        cdf_dm = np.zeros_like(dm_pdf)
        for i in range(len(cdf_dm)):
            cdf_dm[i] = np.trapz(dm_pdf[:i],er_list[:i])
        cdf_dm /= cdf_dm.max()
        dm_events_sim = int(dm_rate * exposure)

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
                print 'Predicted Number of Nu events: {}'.format(int(sum(Nu_events_sim)))
                print 'Predicted Number of DM events: {}'.format(dm_events_sim)

            # Simulate events
            print('Evaluated Events: Neutrino {:.0f}, DM {:.0f}'.format(int(sum(nevts_n)), nevts_dm))
            u = random.rand(nevts_dm)
            # Generalize to rejection sampling algo for time implimentation
            e_sim2 = np.zeros(nevts_dm)
            for i in range(nevts_dm):
                 e_sim2[i] = er_list[np.absolute(cdf_dm - u[i]).argmin()]

            e_sim = np.concatenate((e_sim, e_sim2))
            times = np.zeros_like(e_sim)

            if not QUIET:
                print 'Running Likelihood Analysis...'
            # Minimize likelihood -- MAKE SURE THIS MINIMIZATION DOESNT FAIL. CONSIDER USING GRADIENT INFO
            nu_bnds = [(-5.0, 3.0)] * nu_contrib
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
            #print 'DM Vals: ', max_dm
            #print 'No DM: ', max_nodm
            #print like_init_dm.test_num_events(max_dm.x[:-1], max_dm.x[-1])
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
        print tstat_arr

        print 'FINISHED CYCLE'
        print 'True DM mass: ', mass
        print 'True DM sigma_p: ', sigmap
        #print("--- %s seconds ---" % (time.time() - start_time))


        if len(tstat_arr) > 0:
            print 'Median Q: {:.2f}'.format(np.median(tstat_arr))
            print 'Mean Q: {:.2f}\n'.format(np.mean(tstat_arr))
            print 'T-stat Array:', tstat_arr
            testqsimp = float(np.sum(tstat_arr > q_goal)) / float(len(tstat_arr))
            if testqsimp == 1.:
                testq = 1.
            else:
                #testq = float(np.sum(tstat_arr > q_goal)) / float(len(tstat_arr))
                kernel = gaussian_kde(tstat_arr)
                xprob = np.linspace(0., np.max(tstat_arr), 200)
                testq = np.trapz(kernel(xprob[xprob > q_goal]), xprob[xprob > q_goal]) / \
                        np.trapz(kernel(xprob),xprob)
            sig_list.append([sig, testq])
            sig_list.sort(key=lambda x: x[0])

            print 'testq (mean, end n cycle): {}'.format(testq)

            print 'testq: {} --> WRITE'.format(testq)
            print '~~~~~~~~~~~~~~~~~~~~~MOVING ON~~~~~~~~~~~~~~~~~~~~~'
            print '\n\n'
            if os.path.exists(file_info):
                load_old = np.loadtxt(file_info)
                new_arr = np.vstack((load_old, np.array([np.log10(sigmap), testq])))
                new_arr = new_arr[new_arr[:, 0].argsort()]
                np.savetxt(file_info, new_arr)
            else:
                np.savetxt(file_info, np.array([np.log10(sigmap), testq]))


    return
