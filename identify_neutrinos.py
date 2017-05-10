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
Sv_dir = path + '/NeutrinoSims/'


def identify_nu(exposure_low=1., exposure_high=100., expose_num=30, element='Germanium',
                file_tag='', n_runs=20, Eth='', Ehigh=6., identify=np.array(['reactor']),
                red_uncer=1.):

    exposure_list = np.logspace(np.log10(exposure_low), np.log10(exposure_high), expose_num)

    sim_files_exist = True
    file_info = path + '/Saved_Files/'

    print 'Run Info:'
    print 'Experiment: ', element
    print 'Identifying {} species: '.format(len(identify)), identify
    print '\n'

    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenLAB)
    if Eth > 0:
        Qmin = Eth
    Qmax = Ehigh
    print 'Qmin: {:.2f}, Qmax: {:.2f}'.format(Qmin, Qmax)

    file_info += 'Identifying_'
    for nu in identify:
        file_info += nu + '_'
    file_info += element + '_Eth_{:.2f}_'.format(Qmin) + labor + '_' + file_tag
    file_info += 'Reduce_Err_{:.2e}.dat'.format(red_uncer)

    print 'Output File: ', file_info
    print '\n'


    # make sure there are enough points for numerical accuracy/stability
    er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), 300)

    nu_comp = ['b8', 'b7l1', 'b7l2', 'pepl1', 'hep', 'pp', 'o15', 'n13', 'f17',
               'reactor', 'geoU', 'geoTh', 'geoK']

    keep_nus = []
    for i in range(len(nu_comp)):
        if nu_comp[i] in identify:
            if Nu_spec(Nu_spec).max_er_from_nu(NEUTRINO_EMAX[nu_comp[i]], experiment_info[0][0]) < Qmin:
                print 'Threshold too low for ', nu_comp[i]
                print 'Exiting...'
                exit()
            else:
                continue
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

    nuspecLOOK = np.zeros(len(identify), dtype=object)
    nu_rateLOOK = np.zeros(len(identify), dtype=object)
    nu_pdfLOOK = np.zeros(len(identify), dtype=object)
    cdf_nuLOOK = np.zeros(len(identify), dtype=object)
    Nu_events_simLOOK = np.zeros(len(identify))

    nu_events = np.zeros(nu_contrib, dtype=object)
    nu_eventsLOOK = np.zeros(len(identify), dtype=object)
    for i in range(nu_contrib):
        nuspec[i] = np.zeros_like(er_list)
        try:
            nu_sim = Sv_dir + 'Simulate_' + nu_comp[i] + '_' + element
            nu_sim += '_Eth_{:.2f}_Emax_{:.2f}_'.format(Qmin, Qmax) + labor + '_'
            nu_sim += file_tag + '.dat'
            nu_events[i] = np.loadtxt(nu_sim)
        except IOError:
            print 'No pre-simulated files...'
            print nu_sim
            sim_files_exist = False


    for i in range(len(identify)):
        nuspecLOOK[i] = np.zeros_like(er_list)
        try:
            nu_sim = Sv_dir + 'Simulate_' + identify[i] + '_' + element
            nu_sim += '_Eth_{:.2f}_Emax_{:.2f}_'.format(Qmin, Qmax) + labor + '_'
            nu_sim += file_tag + '.dat'
            nu_eventsLOOK[i] = np.loadtxt(nu_sim)
        except IOError:
            print 'No pre-simulated files...'
            sim_files_exist = False

    for iso in experiment_info:
        for i in range(nu_contrib):
            nuspec[i] += Nu_spec(labor).nu_rate(nu_comp[i], er_list, iso)
        for i in range(len(identify)):
            nuspecLOOK[i] += Nu_spec(labor).nu_rate(identify[i], er_list, iso)

    for i, MT in enumerate(exposure_list):
        tstat_arr = np.zeros(n_runs)
        print 'Exposure, ', MT

        try:
            check = np.loadtxt(file_info)
            try:
                if MT in check[:, 0]:
                    continue
            except IndexError:
                if MT == check[0]:
                    continue
        except IOError:
            pass

        for i in range(nu_contrib):
            nu_rate[i] = np.trapz(nuspec[i], er_list)
            print nu_comp[i], nu_rate[i]*MT
            if nu_rate[i] > 0.:
                nu_pdf[i] = nuspec[i] / nu_rate[i]
                cdf_nu[i] = np.zeros_like(nu_pdf[i])
                for j in range(len(nu_pdf[i])):
                    cdf_nu[i][j] = np.trapz(nu_pdf[i][:j], er_list[:j])

                cdf_nu[i] /= cdf_nu[i].max()
                Nu_events_sim[i] = int(nu_rate[i] * MT)

        for i in range(len(identify)):
            nu_rateLOOK[i] = np.trapz(nuspecLOOK[i], er_list)
            print identify[i], nu_rateLOOK[i]*MT
            if nu_rateLOOK[i] > 0.:
                nu_pdfLOOK[i] = nuspecLOOK[i] / nu_rateLOOK[i]
                cdf_nuLOOK[i] = np.zeros_like(nu_pdfLOOK[i])
                for j in range(len(nu_pdfLOOK[i])):
                    cdf_nuLOOK[i][j] = np.trapz(nu_pdfLOOK[i][:j],er_list[:j])

                cdf_nuLOOK[i] /= cdf_nuLOOK[i].max()
                Nu_events_simLOOK[i] = int(nu_rateLOOK[i] * MT)

        nevts_n = np.zeros(nu_contrib, dtype='int')
        nevts_nLOOK = np.zeros(len(identify), dtype='int')

        fails = np.array([])
        nn = 0
        while nn < n_runs:
            print 'Run {:.0f} of {:.0f}'.format(nn + 1, n_runs)
            for i in range(nu_contrib):
                if nu_comp[i] == "reactor":
                    nu_mean_f, nu_sig = reactor_flux(loc=labor)
                elif "geo" in nu_comp[i]:
                    nu_mean_f, nu_sig = geo_flux(loc=labor, el=nu_comp[i][3:])
                else:
                    nu_sig = NEUTRINO_SIG[nu_comp[i]]
                    nu_mean_f = NEUTRINO_MEANF[nu_comp[i]]
                nu_sig *= red_uncer
                nerr = nu_sig / nu_mean_f * Nu_events_sim[i]
                try:
                    # nevts_n[i] = poisson.rvs(int(Nu_events_sim[i]))
                    nevts_n[i] = random.normal(loc=Nu_events_sim[i], scale=nerr)
                    if nevts_n[i] < 0:
                        nevts_n[i] = 0
                    #print nu_comp[i], nevts_n[i], Nu_events_sim[i], nerr
                except ValueError:
                    nevts_n[i] = 0
            for i in range(len(identify)):
                if identify[i] == "reactor":
                    nu_mean_f, nu_sig = reactor_flux(loc=labor)
                elif "geo" in identify[i]:
                    nu_mean_f, nu_sig = geo_flux(loc=labor, el=identify[i][3:])
                else:
                    nu_sig = NEUTRINO_SIG[identify[i]]
                    nu_mean_f = NEUTRINO_MEANF[identify[i]]
                nu_sig *= red_uncer
                nerr = nu_sig / nu_mean_f * Nu_events_simLOOK[i]
                try:
                    # nevts_nLOOK[i] = poisson.rvs(int(Nu_events_simLOOK[i]))
                    nevts_nLOOK[i] = random.normal(loc=Nu_events_simLOOK[i], scale=nerr)
                    if nevts_nLOOK[i] < 0:
                        nevts_nLOOK[i] = 0
                    #print 'Identifying: ', identify[i], nevts_nLOOK[i], Nu_events_simLOOK[i], nerr
                except ValueError:
                    nevts_nLOOK[i] = 0

            if not QUIET:
                print 'Predicted Number of Nu BKG events: {}'.format(int(sum(Nu_events_sim)))
                print 'Predicted Number of Nu of interest events: {}'.format(Nu_events_simLOOK)

            Nevents = int(sum(nevts_n))
            NeventsLOOK = int(sum(nevts_nLOOK))

            if sim_files_exist:
                e_sim = np.array([])
                e_simLOOK = np.array([])
                for i in range(nu_contrib):
                    u = random.rand(nevts_n[i]) * len(nu_events[i])
                    e_sim = np.append(e_sim, nu_events[i][u.astype(int)])
                for i in range(len(identify)):
                    u = random.rand(nevts_nLOOK[i]) * len(nu_eventsLOOK[i])
                    e_simLOOK = np.append(e_simLOOK, nu_eventsLOOK[i][u.astype(int)])

            else:
                u = random.rand(Nevents)
                e_sim = np.zeros(Nevents)
                sum_nu_evts = np.zeros(nu_contrib)

                uLOOK = random.rand(NeventsLOOK)
                e_simLOOK = np.zeros(NeventsLOOK)
                sum_nu_evtsLOOK = np.zeros(len(identify))

                for i in range(nu_contrib):
                    sum_nu_evts[i] = np.sum(nevts_n[:i + 1])
                sum_nu_evts = np.insert(sum_nu_evts, 0, -1)

                for i in range(len(identify)):
                    sum_nu_evtsLOOK[i] = np.sum(nevts_nLOOK[:i + 1])
                sum_nu_evtsLOOK = np.insert(sum_nu_evtsLOOK, 0, -1)


                for i in range(Nevents):
                    if i < sum(nevts_n):
                        for j in range(nu_contrib + 1):
                            if sum_nu_evts[j] <= i < sum_nu_evts[j + 1]:
                                e_sim[i] = er_list[np.absolute(cdf_nu[j] - u[i]).argmin()]

                for i in range(NeventsLOOK):
                    if i < sum(nevts_nLOOK):
                        for j in range(len(identify) + 1):
                            if sum_nu_evtsLOOK[j] <= i < sum_nu_evtsLOOK[j + 1]:
                                e_simLOOK[i] = er_list[np.absolute(cdf_nuLOOK[j] - uLOOK[i]).argmin()]

            e_sim = np.concatenate((e_sim, e_simLOOK))

            # Simulate events
            print('Evaluated Events: Neutrino {:.0f}, NOI {:.0f}'.format(int(sum(nevts_n)), int(sum(nevts_nLOOK))))

            if not QUIET:
                print 'Running Likelihood Analysis...'

            nu_bnds = [(-20.0, 3.0)] * nu_contrib
            full_bnds = [(-20.0, 3.0)] * (nu_contrib + len(identify))
            print 'Just Background...'
            like_init_bkg = Likelihood_analysis('sigma_si', 'fnfp_si', 10., 0., 1.,
                                                 MT, element, experiment_info,
                                                 e_sim, np.zeros_like(e_sim), nu_comp, labor,
                                                 nu_contrib, Qmin, Qmax, reduce_uncer=red_uncer, DARK=False)

            max_bkg = minimize(like_init_bkg.likelihood, np.zeros(nu_contrib),
                                args=(np.array([-100.])), tol=1e-4, method='SLSQP',
                                options={'maxiter': 100}, bounds=nu_bnds,
                                jac=like_init_bkg.like_gradi)


            print 'Now With IDENTIFY...'

            like_init_tot = Likelihood_analysis('sigma_si', 'fnfp_si', 10., 0., 1.,
                                               MT, element, experiment_info, e_sim, np.zeros_like(e_sim),
                                               np.concatenate((nu_comp, identify)), labor,
                                               (nu_contrib+len(identify)),
                                               Qmin, Qmax, reduce_uncer=red_uncer, DARK=False)

            max_tot = minimize(like_init_tot.likelihood, np.zeros(nu_contrib + len(identify)),
                               args=(np.array([-100.]), range(nu_contrib, nu_contrib + len(identify))),
                               tol=1e-4, method='SLSQP',
                               options={'maxiter': 100}, bounds=full_bnds,
                               jac=like_init_tot.like_gradi)

            print 'Minimizaiton Success: ', max_bkg.success, max_tot.success
            print 'Values: ', max_bkg.fun, max_tot.fun
            print max_bkg
            print max_tot
            #print like_init_bkg.likelihood(max_tot.x[:nu_contrib], np.array([-100.]))

            if not max_bkg.success or not max_tot.success:
                fails = np.append(fails, nn)

            test_stat = np.max([max_bkg.fun - max_tot.fun, 0.])

            pval = chi2.sf(test_stat, len(identify))

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
        # print("--- %s seconds ---" % (time.time() - start_time))


        if len(tstat_arr) > 0:
            print 'Median Q: {:.2f}'.format(np.median(tstat_arr))
            print 'Mean Q: {:.2f}\n'.format(np.mean(tstat_arr))
            print 'T-stat Array:', tstat_arr

            #testqsimp = float(np.sum(tstat_arr > q_goal)) / float(len(tstat_arr))

            if np.all(tstat_arr == 0.):
                testq = 0.
            else:
                kernel = gaussian_kde(tstat_arr)
                xprob = np.logspace(-3, np.log10(np.max(tstat_arr)), 50)
                norm = np.trapz(kernel(xprob), xprob)
                #testq = np.trapz(kernel(xprob[xprob > q_goal]), xprob[xprob > q_goal]) / norm
                #print np.column_stack((xprob, kernel(xprob)/norm))

                check_if0 = np.trapz(kernel(xprob[xprob > 1e-3]), xprob[xprob > 1e-3]) / norm

                if check_if0 < 0.9:
                    testq = 0.
                else:
                    look_for_90 = np.zeros_like(xprob)
                    for i in range(len(xprob)):
                        look_for_90[i] = np.trapz(kernel(xprob[i:]), xprob[i:]) / norm
                    #print np.column_stack((xprob, look_for_90))
                    testq = xprob[np.argmin(np.abs(look_for_90 - 0.9))]
            print 'testq (mean, end n cycle): {}'.format(testq)

            print 'testq: {} --> WRITE'.format(testq)
            print '~~~~~~~~~~~~~~~~~~~~~MOVING ON~~~~~~~~~~~~~~~~~~~~~'
            print '\n\n'
            if os.path.exists(file_info):
                load_old = np.loadtxt(file_info)
                new_arr = np.vstack((load_old, np.array([MT, testq])))
                new_arr = new_arr[new_arr[:, 0].argsort()]
                np.savetxt(file_info, new_arr)
            else:
                np.savetxt(file_info, np.array([MT, testq]))

    return



def identify_nu_naieve(exposure_low=1., exposure_high=100., expose_num=20, element='Germanium',
                       file_tag='', n_runs=20, Eth=-1, identify=np.array(['geoU', 'geoTh', 'geoK']),
                       red_uncer=1.):
    exposure_list = np.logspace(np.log10(exposure_low), np.log10(exposure_high), expose_num)

    print 'Run Info:'
    print 'Experiment: ', element
    print 'Identifying {} species: '.format(len(identify)), identify
    print '\n'

    maxE = 0.
    for i in identify:
        elem, Qmax, Qmin = Element_Info(element)
        maxER = Nu_spec(lab='Snolab').max_er_from_nu(NEUTRINO_EMAX[i], elem[0, 0])
        if maxER > maxE:
            maxE = maxER

    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenLAB)
    if Eth > 0:
        Qmin = Eth
    Qmax = maxE
    print 'Qmin: {:.2f}, Qmax: {:.2f}'.format(Qmin, Qmax)
    er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), 300)
    nu_comp = ['b8', 'b7l1', 'b7l2', 'pepl1', 'hep', 'pp', 'o15', 'n13', 'f17',
               'reactor', 'geoU', 'geoTh', 'geoK']

    keep_nus = []
    for i in range(len(nu_comp)):
        if nu_comp[i] in identify:
            if Nu_spec(Nu_spec).max_er_from_nu(NEUTRINO_EMAX[nu_comp[i]], experiment_info[0][0]) < Qmin:
                print 'Threshold too low for ', nu_comp[i]
                print 'Exiting...'
                exit()
            else:
                continue
        if Nu_spec(Nu_spec).max_er_from_nu(NEUTRINO_EMAX[nu_comp[i]], experiment_info[0][0]) > Qmin:
            keep_nus.append(i)
    nu_comp = [x for i, x in enumerate(nu_comp) if i in keep_nus]
    nu_contrib = len(nu_comp)

    print 'Neutrinos Considered: ', nu_comp

    nuspec = np.zeros(nu_contrib, dtype=object)
    nu_rate = np.zeros(nu_contrib, dtype=object)
    nuspecLOOK = np.zeros(len(identify), dtype=object)
    nu_rateLOOK = np.zeros(len(identify), dtype=object)

    for i in range(nu_contrib):
        nuspec[i] = np.zeros_like(er_list)
    for i in range(len(identify)):
        nuspecLOOK[i] = np.zeros_like(er_list)

    for iso in experiment_info:
        for i in range(nu_contrib):
            nuspec[i] += Nu_spec(labor).nu_rate(nu_comp[i], er_list, iso)
        for i in range(len(identify)):
            nuspecLOOK[i] += Nu_spec(labor).nu_rate(identify[i], er_list, iso)

    for i, MT in enumerate(exposure_list):

        print 'Exposure, ', MT

        for i in range(nu_contrib):
            nu_rate[i] = MT * np.trapz(nuspec[i], er_list)
        for i in range(len(identify)):
            nu_rateLOOK[i] = MT * np.trapz(nuspecLOOK[i], er_list)

        print 'Nbkg {:.0f}, Nidentify {:.0f}'.format(np.sum(nu_rate), np.sum(nu_rateLOOK))
        nerr = 0.
        # for i in range(nu_contrib):
        #     if nu_comp[i] == "reactor":
        #         nu_mean_f, nu_sig = reactor_flux(loc=labor)
        #     elif "geo" in nu_comp[i]:
        #         nu_mean_f, nu_sig = geo_flux(loc=labor, el=nu_comp[i][3:])
        #     else:
        #         nu_sig = NEUTRINO_SIG[nu_comp[i]]
        #         nu_mean_f = NEUTRINO_MEANF[nu_comp[i]]
        #     nu_sig *= red_uncer
        #     nerr += nu_sig / nu_mean_f * nu_rate[i]
        for i in range(nu_contrib):
            nerr += np.sqrt(nu_rate[i])

        print 'Exposure {:.2f}, Significance {:.3f}'.format(MT, np.sum(nu_rateLOOK)/nerr)
    return