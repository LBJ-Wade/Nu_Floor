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
from scipy.optimize import fsolve

path = os.getcwd()
Sv_dir = path + '/NeutrinoSims/'
QUIET = False
xenLAB = 'LZ'


def nu_floor(sig_low, sig_high, n_sigs=10, model="sigma_si", mass=6., fnfp=1.,
             element='Germanium', exposure=1., delta=0., GF=False, time_info=False,
             file_tag='', n_runs=20, Eth=''):

    #start_time = time.time()
    testq = 0
    sim_files_exist = True
    sim_dm_file_exist = True
    file_info = path + '/Saved_Files/'

    print 'Run Info:'
    print 'Experiment: ', element
    print 'Model: ', model
    coupling = "fnfp" + model[5:]
    print 'Coupling: ', coupling, fnfp
    print 'Mass: {:.2f}'.format(mass)

    print '\n'

    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenLAB)
    if Eth > 0:
        Qmin = Eth

    file_info += element + '_' + model + '_' + coupling + '_{:.2f}'.format(fnfp)
    file_info += '_Exposure_{:.2f}_tonyr_DM_Mass_{:.2f}_GeV'.format(exposure, mass)
    file_info += '_Eth_{:.2f}_'.format(Qmin) + labor + '_delta_{:.2f}'.format(delta)
    file_info += file_tag + '.dat'
    print 'Output File: ', file_info
    print '\n'


    mindm = np.zeros(len(experiment_info[:, 0]))
    for i, iso in enumerate(experiment_info):
        mindm[i] = MinDMMass(iso[0], delta, Qmin, vesc=533.+232.)
    MinDM = np.min(mindm)
    print 'Minimum DM mass: ', MinDM
    if mass < MinDM + 0.5:
        print 'Mass too small...'
        return
    # 3\sigma for Chi-square Dist with 1 DoF means q = 9.0
    q_goal = 9.0

    # make sure there are enough points for numerical accuracy/stability


    nu_comp = ['b8', 'b7l1', 'b7l2', 'pepl1', 'hep', 'pp', 'o15', 'n13', 'f17', 'atm',
                'dsnb3mev', 'dsnb5mev', 'dsnb8mev', 'reactor', 'geoU', 'geoTh', 'geoK']

    keep_nus = []
    max_es = np.zeros(len(nu_comp))
    for i in range(len(nu_comp)):
        max_es[i] = Nu_spec(labor).max_er_from_nu(NEUTRINO_EMAX[nu_comp[i]], experiment_info[0][0])
        if max_es[i] > Qmin:
            keep_nus.append(i)
            if max_es[i] > Qmax:
                max_es[i] = Qmax

    nu_comp = [x for i,x in enumerate(nu_comp) if i in keep_nus]
    max_es = max_es[keep_nus]
    nu_contrib = len(nu_comp)
    print 'Neutrinos Considered: ', nu_comp
    NERG = 300
    er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), NERG)
    time_list = np.zeros_like(er_list)
    er_nu = np.zeros(nu_contrib, dtype=object)

    nuspec = np.zeros(nu_contrib, dtype=object)
    nu_rate = np.zeros(nu_contrib, dtype=object)
    Nu_events_sim = np.zeros(nu_contrib)

    nu_events = np.zeros(nu_contrib, dtype=object)
    for i in range(nu_contrib):
        nuspec[i] = np.zeros_like(er_list)
        try:
            nu_sim = Sv_dir + 'Simulate_' + nu_comp[i] + '_' + element
            nu_sim += '_Eth_{:.2f}_Emax_{:.2f}_'.format(Qmin, Qmax) + labor + '_'
            nu_sim += '_.dat'
            nu_events[i] = np.loadtxt(nu_sim)
        except IOError:
            print 'No pre-simulated files...'
            sim_files_exist = False
            exit()

    for i in range(nu_contrib):
        er_nu[i] = np.logspace(np.log10(Qmin), np.log10(max_es[i]), NERG)
        for iso in experiment_info:
            nuspec[i] += Nu_spec(labor).nu_rate(nu_comp[i], er_nu[i], iso)
        nu_rate[i] = np.trapz(nuspec[i], er_nu[i])
        print nu_comp[i], nu_rate[i]

        if nu_rate[i] > 0.:
            Nu_events_sim[i] = int(nu_rate[i] * exposure)

    nevts_n = np.zeros(nu_contrib, dtype='int')
    print '\n \n'

    dm_recoil_sv = path + '/DarkMatterSims/Simulate_DarkMatter_' + element
    dm_recoil_sv += '_' + model + '_' + coupling + '_{:.2f}_DM_Mass_{:.2f}_GeV'.format(fnfp, mass)
    dm_recoil_sv += '_Eth_{:.2f}_Emax_{:.2f}_'.format(Qmin, Qmax) + labor + '_'
    if delta != 0.:
        dm_recoil_sv += 'delta_{:.2f}_'.format(delta)
    dm_recoil_sv += file_tag + '.dat'
    try:
        dm_recoil_list = np.loadtxt(dm_recoil_sv)
    except IOError:
        print 'Dark Matter Recoils Not Saved...'
        sim_dm_file_exist = False

    drdq_params = default_rate_parameters.copy()
    drdq_params['element'] = element
    drdq_params['mass'] = mass
    drdq_params[coupling] = fnfp
    drdq_params['delta'] = delta
    drdq_params['GF'] = GF
    drdq_params['time_info'] = time_info
    arbitrary_norm = 1e-40
    drdq_params[model] = arbitrary_norm

    dm_rate = R(Qmin=Qmin, Qmax=Qmax, **drdq_params) * 10. ** 3. * s_to_yr

    win = False
    sig_list = []
    while not win:
        print 'Sigma List: ', sig_list
        sig, win = adaptive_samples(np.log10(sig_low), np.log10(sig_high), sig_list)
        sigmap = 10.**sig
        if len(sig_list) > n_sigs:
            win = True
            continue

        print 'Sigma: {:.2e}'.format(sigmap)

        try:
            check = np.loadtxt(file_info)
            try:
                if np.log10(sigmap) in check[:, 0]:
                    print 'Check', check
                    for i in range(len(check[:,0])):
                        sig_list.append([check[i,0], check[i,1]])
                    continue
            except IndexError:
                if np.log10(sigmap) == check[0]:
                    print 'Check', check
                    sig_list.append([check[0], check[1]])
                    continue

        except IOError:
            pass

        dm_events_sim = int(dm_rate * exposure * (sigmap / arbitrary_norm))

        tstat_arr = np.zeros(n_runs)
        nn = 0

        fails = np.array([])
        while nn < n_runs:
            for i in range(nu_contrib):
                # if nu_comp[i] == "reactor":
                #     nu_mean_f, nu_sig = reactor_flux(loc=labor)
                # elif "geo" in nu_comp[i]:
                #     nu_mean_f, nu_sig = geo_flux(loc=labor, el=nu_comp[i][3:])
                # else:
                #     nu_sig = NEUTRINO_SIG[nu_comp[i]]
                #     nu_mean_f = NEUTRINO_MEANF[nu_comp[i]]
                #
                # nerr = nu_sig / nu_mean_f * Nu_events_sim[i]
                try:
                    nevts_n[i] = poisson.rvs(int(Nu_events_sim[i]))
                    # nevts_n[i] = random.normal(loc=Nu_events_sim[i], scale=nerr)
                    # if nevts_n[i] < 0:
                    #     nevts_n[i] = 0
                except ValueError:
                    nevts_n[i] = 0

            Nevents = int(sum(nevts_n))

            if sim_files_exist:
                e_sim = np.array([])
                for i in range(nu_contrib):
                    u = random.rand(nevts_n[i]) * len(nu_events[i])
                    e_sim = np.append(e_sim, nu_events[i][u.astype(int)])

            else:
                u = random.rand(Nevents)
                e_sim = np.zeros(Nevents)
                sum_nu_evts = np.zeros(nu_contrib)
                for i in range(nu_contrib):
                    sum_nu_evts[i] = np.sum(nevts_n[:i + 1])
                sum_nu_evts = np.insert(sum_nu_evts, 0, -1)

                for i in range(Nevents):
                    if i < sum(nevts_n):
                        for j in range(nu_contrib + 1):
                            if sum_nu_evts[j] <= i < sum_nu_evts[j + 1]:
                                e_sim[i] = er_nu[i][np.absolute(cdf_nu[j] - u[i]).argmin()]

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
            if sim_dm_file_exist:
                e_sim2 = np.array([])
                u = random.rand(nevts_dm) * len(dm_recoil_list)
                e_sim2 = np.append(e_sim2, dm_recoil_list[u.astype(int)])
            else:
                drdq_params[model] = sigmap
                dm_spec = dRdQ(er_list, time_list, **drdq_params) * 10. ** 3. * s_to_yr
                dm_pdf = dm_spec / dm_rate
                cdf_dm = np.zeros_like(dm_pdf)
                for i in range(len(cdf_dm)):
                    cdf_dm[i] = np.trapz(dm_pdf[:i], er_list[:i])
                cdf_dm /= cdf_dm.max()

                u = random.rand(nevts_dm)
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

            # if nn > 1:
            #     print("--- %s seconds ---" % (time.time() - start_time))
            # start_time = time.time()
            like_init_nodm = Likelihood_analysis(model, coupling, mass, 0., fnfp,
                                                 exposure, element, experiment_info,
                                                 e_sim, times, nu_comp, labor,
                                                 nu_contrib, er_nu, nuspec, nu_rate,
                                                 Qmin, Qmax, delta=delta, time_info=time_info,
                                                 GF=False, DARK=False)

            max_nodm = minimize(like_init_nodm.likelihood, np.zeros(nu_contrib),
                                args=(np.array([-100.])), tol=1e-2, method='SLSQP',
                                options={'maxiter': 100}, bounds=nu_bnds,
                                jac=like_init_nodm.like_gradi)

            like_init_dm = Likelihood_analysis(model, coupling, mass, 1., fnfp,
                                               exposure, element, experiment_info, e_sim, times, nu_comp, labor,
                                               nu_contrib, er_nu, nuspec, nu_rate,
                                               Qmin, Qmax, delta, time_info=time_info, GF=False)

            max_dm = minimize(like_init_dm.like_multi_wrapper,
                              np.concatenate((np.zeros(nu_contrib), np.array([np.log10(sigmap)]))),
                              tol=1e-2, method='SLSQP', bounds=dm_bnds,
                              options={'maxiter': 100}, jac=like_init_dm.likegrad_multi_wrapper)

            #print 'DM Vals: ', max_dm
            #print 'No DM: ', max_nodm


            #print like_init_dm.test_num_events(max_dm.x[:-1], max_dm.x[-1])

            print 'Minimizaiton Success: ', max_nodm.success, max_dm.success
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

        # mask = np.array([(i in fails) for i in xrange(len(tstat_arr))])
        # tstat_arr = tstat_arr[~mask]
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
            elif np.all(tstat_arr == 0.):
                testq = 0.
            else:
                # testq = float(np.sum(tstat_arr > q_goal)) / float(len(tstat_arr))
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


def nu_floor_Bound(sig_low, sig_high, n_sigs=10,
             model="sigma_si", mass=6., fnfp=1.,
             element='Germanium', exposure=1., delta=0., GF=False, time_info=False,
             file_tag='', n_runs=20, Eth=''):

    #start_time = time.time()
    testq = 0.
    sim_files_exist = True
    sim_dm_file_exist = True
    file_info = path + '/FutureProjBound/'
    file_info += 'DerivedBound_'

    print 'Run Info:'
    print 'Experiment: ', element
    print 'Model: ', model
    coupling = "fnfp" + model[5:]
    print 'Coupling: ', coupling, fnfp
    print 'Mass: {:.2f}'.format(mass)

    print '\n'

    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenLAB)
    if Eth > 0:
        Qmin = Eth

    file_info += element + '_' + model + '_' + coupling + '_{:.2f}'.format(fnfp)
    file_info += '_Exposure_{:.2f}_tonyr_DM_Mass_{:.2f}_GeV'.format(exposure, mass)
    file_info += '_Eth_{:.2f}_'.format(Qmin) + labor + '_delta_{:.2f}'.format(delta)
    file_info += file_tag + '.dat'
    print 'Output File: ', file_info
    print '\n'


    mindm = np.zeros(len(experiment_info[:, 0]))
    for i, iso in enumerate(experiment_info):
        mindm[i] = MinDMMass(iso[0], delta, Qmin, vesc=533.+232.)
    MinDM = np.min(mindm)
    print 'Minimum DM mass: ', MinDM
    if mass < MinDM + 0.5:
        print 'Mass too small...'
        return
    # 3\sigma for Chi-square Dist with 1 DoF means q = 9.0
    q_goal = 2.7

    # make sure there are enough points for numerical accuracy/stability


    nu_comp = ['b8', 'b7l1', 'b7l2', 'pepl1', 'hep', 'pp', 'o15', 'n13', 'f17', 'atm',
                'dsnb3mev', 'dsnb5mev', 'dsnb8mev', 'reactor', 'geoU', 'geoTh', 'geoK']

    keep_nus = []
    max_es = np.zeros(len(nu_comp))
    for i in range(len(nu_comp)):
        max_es[i] = Nu_spec(labor).max_er_from_nu(NEUTRINO_EMAX[nu_comp[i]], experiment_info[0][0])
        if max_es[i] > Qmin:
            keep_nus.append(i)
            if max_es[i] > Qmax:
                max_es[i] = Qmax

    nu_comp = [x for i,x in enumerate(nu_comp) if i in keep_nus]
    max_es = max_es[keep_nus]
    nu_contrib = len(nu_comp)
    print 'Neutrinos Considered: ', nu_comp
    NERG = 300
    er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), NERG)
    time_list = np.zeros_like(er_list)
    er_nu = np.zeros(nu_contrib, dtype=object)

    nuspec = np.zeros(nu_contrib, dtype=object)
    nu_rate = np.zeros(nu_contrib, dtype=object)
    Nu_events_sim = np.zeros(nu_contrib)

    nu_events = np.zeros(nu_contrib, dtype=object)
    for i in range(nu_contrib):
        nuspec[i] = np.zeros_like(er_list)
        try:
            nu_sim = Sv_dir + 'Simulate_' + nu_comp[i] + '_' + element
            nu_sim += '_Eth_{:.2f}_Emax_{:.2f}_'.format(Qmin, Qmax) + labor + '_'
            nu_sim += '_.dat'
            nu_events[i] = np.loadtxt(nu_sim)
        except IOError:
            print 'No pre-simulated files...'
            sim_files_exist = False
            exit()

    for i in range(nu_contrib):
        er_nu[i] = np.logspace(np.log10(Qmin), np.log10(max_es[i]), NERG)
        for iso in experiment_info:
            nuspec[i] += Nu_spec(labor).nu_rate(nu_comp[i], er_nu[i], iso)
        nu_rate[i] = np.trapz(nuspec[i], er_nu[i])
        print nu_comp[i], nu_rate[i]

        if nu_rate[i] > 0.:
            Nu_events_sim[i] = int(nu_rate[i] * exposure)

    nevts_n = np.zeros(nu_contrib, dtype='int')
    print '\n \n'

    drdq_params = default_rate_parameters.copy()
    drdq_params['element'] = element
    drdq_params['mass'] = mass
    drdq_params[coupling] = fnfp
    drdq_params['delta'] = delta
    drdq_params['GF'] = GF
    drdq_params['time_info'] = time_info
    arbitrary_norm = 1e-40
    drdq_params[model] = arbitrary_norm

    dm_rate = 0.
    dm_events_sim = 0.

    tstat_arr = np.zeros(n_runs)
    nn = 0
    
    fails = np.array([])
    sigLIST = []
    while nn < n_runs:
        for i in range(nu_contrib):
            try:
                nevts_n[i] = poisson.rvs(int(Nu_events_sim[i]))
            except ValueError:
                nevts_n[i] = 0

        Nevents = int(sum(nevts_n))

        if sim_files_exist:
            e_sim = np.array([])
            for i in range(nu_contrib):
                u = random.rand(nevts_n[i]) * len(nu_events[i])
                e_sim = np.append(e_sim, nu_events[i][u.astype(int)])

        else:
            u = random.rand(Nevents)
            e_sim = np.zeros(Nevents)
            sum_nu_evts = np.zeros(nu_contrib)
            for i in range(nu_contrib):
                sum_nu_evts[i] = np.sum(nevts_n[:i + 1])
            sum_nu_evts = np.insert(sum_nu_evts, 0, -1)

            for i in range(Nevents):
                if i < sum(nevts_n):
                    for j in range(nu_contrib + 1):
                        if sum_nu_evts[j] <= i < sum_nu_evts[j + 1]:
                            e_sim[i] = er_nu[i][np.absolute(cdf_nu[j] - u[i]).argmin()]

        print 'Run {:.0f} of {:.0f}'.format(nn + 1, n_runs)
        
        nevts_dm = 0

        if not QUIET:
            print 'Predicted Number of Nu events: {}'.format(int(sum(Nu_events_sim)))

        # Simulate events
        print('Evaluated Events: Neutrino {:.0f}'.format(int(sum(nevts_n))))
        times = np.zeros_like(e_sim)

        if not QUIET:
            print 'Running Likelihood Analysis...'
        # Minimize likelihood -- MAKE SURE THIS MINIMIZATION DOESNT FAIL. CONSIDER USING GRADIENT INFO
        nu_bnds = [(-5.0, 3.0)] * nu_contrib
        dm_bnds = nu_bnds + [(-60., -30.)]

        # if nn > 1:
        #     print("--- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
#        like_init_nodm = Likelihood_analysis(model, coupling, mass, 0., fnfp,
#                                             exposure, element, experiment_info,
#                                             e_sim, times, nu_comp, labor,
#                                             nu_contrib, er_nu, nuspec, nu_rate,
#                                             Qmin, Qmax, delta=delta, time_info=time_info,
#                                             GF=False, DARK=False)
#
#        max_nodm = minimize(like_init_nodm.likelihood, np.zeros(nu_contrib),
#                            args=(np.array([-100.]), [], True), tol=1e-2, method='SLSQP',
#                            options={'maxiter': 100}, bounds=nu_bnds,
#                            jac=like_init_nodm.like_gradi)


        like_init_dm = Likelihood_analysis(model, coupling, mass, 1., fnfp,
                                           exposure, element, experiment_info, e_sim, times, nu_comp, labor,
                                           nu_contrib, er_nu, nuspec, nu_rate,
                                           Qmin, Qmax, delta, time_info=time_info, GF=False)

        max_dm = minimize(like_init_dm.like_nu_bound, [-60], args=(np.zeros(nu_contrib)), tol=1e-5,
                          bounds=[(-70, -30)], options={'maxiter': 100})
#        max_dm = minimize(like_init_dm.like_nu_bound,
#                          np.concatenate((np.zeros(nu_contrib), np.array([np.log10(1e-45)]))),
#                          args=(max_nodm.fun), tol=1e-4, method='SLSQP', bounds=dm_bnds,
#                          options={'maxiter': 100})#, jac=like_init_dm.like_nu_bnd_jac)
        try:
            bnd = fsolve(lambda x: like_init_dm.like_nu_bound(x, np.zeros(nu_contrib)) - max_dm.fun - 2.7, -43.)
        except:
            continue
        
        
        #print R(Qmin=Qmin, Qmax=Qmax, **drdq_params) * 10. ** 3. * s_to_yr
        
        #print 'Minimizaiton Success: ', max_nodm.success, max_dm.success
#        if not max_nodm.success or not max_dm.success:
#            fails = np.append(fails, nn)
#            continue

        #sigLIM = max_dm.x[-1]
        sigLIST.append(bnd)
        nn += 1
    
    sigLIST.sort()
    totpts = float(len(sigLIST))
    keyval = int(0.9*totpts)
    
    sigLIST.sort()
    sigVAL =sigLIST[keyval]
    #print sigVAL, file_info
    np.savetxt(file_info, [sigVAL])
    return
