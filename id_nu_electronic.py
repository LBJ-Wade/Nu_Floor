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
from experiments import *

path = os.getcwd()
QUIET = False
xenLAB = 'LZ'
Sv_dir = path + '/NeutrinoSims/'


def identify_nu_electronic(exposure_low=1., exposure_high=100., expose_num=30, element='Germanium',
                file_tag='', n_runs=20, Eth='', Ehigh=6., identify=np.array(['reactor']),
                red_uncer=1.):

    exposure_list = np.logspace(np.log10(exposure_low), np.log10(exposure_high), expose_num)

    sim_files_exist = True
    file_info = path + '/Saved_Files/'
    file_infoNR = path + '/Saved_Files/'

    print 'Run Info:'
    print 'Experiment: ', element
    print 'Identifying {} species: '.format(len(identify)), identify
    print 'ELECTRONIC ANALYSIS...'
    print '\n'

    # Load experiment info
    experiment_info, Qmin, Qmax = Element_Info(element, electronic=True)
    experiment_info_NR, Qmin_NR, Qmax_NR = Element_Info(element, electronic=False)
    labor = laboratory(element, xen=xenLAB)
    if Eth > 0:
        Qmin = Eth
#    Qmax = Ehigh
    print 'Qmin: {:.2f}, Qmax: {:.2f}'.format(Qmin, Qmax)

    file_info += 'Identifying_'
    file_infoNR += 'Identifying_'
    for nu in identify:
        file_info += nu + '_'
        file_infoNR += nu + '_'
    file_info += 'Electronic_'
    file_info += element + '_Eth_{:.2f}_'.format(Qmin) + labor + '_' + file_tag + '.dat'
    file_infoNR += element + '_Eth_{:.2f}_'.format(Qmin_NR) + labor + '_' + file_tag + '.dat'

    print 'Output File: ', file_info
    print '\n'

    # make sure there are enough points for numerical accuracy/stability
    er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), 300)
    er_list_NR = np.logspace(np.log10(Qmin), np.log10(Qmax), 300)

    nu_comp = ['b8', 'b7l1', 'b7l2', 'pepl1', 'hep', 'pp', 'o15', 'n13', 'f17',
               'reactor', 'geoU', 'geoTh', 'geoK']

    # Determine which neutrinos cannot contribute in energy ragne
    keep_nus = []
    keep_nus_NR = []
    max_es = np.zeros(len(nu_comp))
    max_es_NR = np.zeros(len(nu_comp))
    max_e_id = np.zeros(len(identify))
    for i in range(len(nu_comp)):
        if nu_comp[i] in identify:
            kk = np.argwhere(nu_comp[i] == identify)
            max_e_id[kk] = Nu_spec(labor).max_er_from_nu(NEUTRINO_EMAX[nu_comp[i]], 5.11e-4)
            if max_e_id[kk] < Qmin:
                print 'Threshold too low for ', nu_comp[i], max_e_id[kk]
                print 'Exiting...'
                exit()
            else:
                if max_e_id[kk] > Qmax:
                    max_e_id[kk] = Qmax
                continue
        max_es[i] = Nu_spec(labor).max_er_from_nu(NEUTRINO_EMAX[nu_comp[i]], 5.11e-4)
        max_es_NR[i] = Nu_spec(labor).max_er_from_nu(NEUTRINO_EMAX[nu_comp[i]], experiment_info[0][0])
        if max_es[i] > Qmin:
            keep_nus.append(i)
            if max_es[i] > Qmax:
                max_es[i] = Qmax
        if (max_es_NR[i] > Qmin_NR):
            keep_nus_NR.append(i)
            if max_es_NR[i] > Qmax_NR:
                max_es_NR[i] = Qmax_NR

    nu_comp = [x for i,x in enumerate(nu_comp) if i in keep_nus]
    nu_comp_NR = [x for i,x in enumerate(nu_comp) if i in keep_nus_NR]
    max_es = max_es[keep_nus]
    max_es_NR = max_es_NR[keep_nus_NR]
    nu_contrib = len(nu_comp)
    nu_contrib_NR = len(nu_comp_NR)

    #print max_e_id
    print 'Electronic Recoils, Neutrinos Considered: ', nu_comp
    print 'Nuclear Recoils, Neutrinos Considered: ', nu_comp_NR

    nuspec = np.zeros(nu_contrib, dtype=object)
    nu_rate = np.zeros(nu_contrib, dtype=object)
    Nu_events_sim = np.zeros(nu_contrib)
    # NR
    nuspec_NR = np.zeros(nu_contrib_NR, dtype=object)
    nu_rate_NR = np.zeros(nu_contrib_NR, dtype=object)
    Nu_events_sim_NR = np.zeros(nu_contrib_NR)

    nuspecLOOK = np.zeros(len(identify), dtype=object)
    nu_rateLOOK = np.zeros(len(identify), dtype=object)
    Nu_events_simLOOK = np.zeros(len(identify))

    nu_events = np.zeros(nu_contrib, dtype=object)
    nu_eventsLOOK = np.zeros(len(identify), dtype=object)

    NERG = 300
    er_nu = np.zeros(nu_contrib, dtype=object)
    er_nu_id = np.zeros(len(identify), dtype=object)

    bkg_list = bkg_electron_specs(element)
    for i in range(nu_contrib):
        nuspec[i] = np.zeros(NERG)
        try:
            nu_sim = Sv_dir + 'Simulate_Electronic_' + nu_comp[i] + '_' + element
            nu_sim += '_Eth_{:.2f}_Emax_{:.2f}_'.format(Qmin, Qmax) + labor + '_'
            nu_sim += '_.dat'
            nu_events[i] = np.loadtxt(nu_sim)
        except IOError:
            print 'No pre-simulated files...'
            print nu_sim
            sim_files_exist = False

#    for i in range(nu_contrib_NR):
#        nuspec_NR[i] = np.zeros(NERG)
#        try:
#            nu_sim = Sv_dir + 'Simulate_' + nu_comp_NR[i] + '_' + element
#            nu_sim += '_Eth_{:.2f}_Emax_{:.2f}_'.format(Qmin_NR, Qmax_NR) + labor + '_'
#            nu_sim += '_.dat'
#            nu_events_NR[i] = np.loadtxt(nu_sim)
#        except IOError:
#            print 'No pre-simulated files...'
#            print nu_sim
#            sim_files_exist = False

    for i in range(len(identify)):
        nuspecLOOK[i] = np.zeros(NERG)
        try:
            nu_sim = Sv_dir + 'Simulate_Electronic_' + identify[i] + '_' + element
            nu_sim += '_Eth_{:.2f}_Emax_{:.2f}_'.format(Qmin, Qmax) + labor + '_'
            nu_sim += '_.dat'
            nu_eventsLOOK[i] = np.loadtxt(nu_sim)
        except IOError:
            print 'No pre-simulated files...'
            sim_files_exist = False

    nuspecBKG = np.zeros(len(bkg_list), dtype=object)
    for i in range(len(bkg_list)):
        nuspecBKG[i] = np.zeros(NERG)
        try:
            nu_sim = Sv_dir + 'Simulate_Electronic_' + bkg_list[i] + '_' + element
            nu_sim += '_Eth_{:.2f}_Emax_{:.2f}_'.format(Qmin, Qmax) + labor + '_'
            nu_sim += '_.dat'
            nuspecBKG[i] = np.loadtxt(nu_sim)
        except IOError:
            print 'No pre-simulated files...'
            sim_files_exist = False

    # Nuclear recoil calc for uncertainties of flux
    nuspec_NR = np.zeros(nu_contrib, dtype=object)
    nu_rate_NR = np.zeros(nu_contrib, dtype=object)
    Nu_events_NR = np.zeros(nu_contrib)
    for i in range(nu_contrib):
        er_nu[i] = np.logspace(np.log10(Qmin_NR), np.log10(Qmax_NR), NERG)
        for iso in experiment_info:
            nuspec_NR[i] += Nu_spec(labor).nu_rate(nu_comp[i], er_nu[i], iso)
        nu_rate_NR[i] = np.trapz(nuspec_NR[i], er_nu[i])


    # Dont vary poisson noise for NR, but run quick binned likelihood analysis to see sensitivity
    ############

    # Continue with electronic
    for i in range(nu_contrib):
        er_nu[i] = np.logspace(np.log10(Qmin), np.log10(max_es[i]), NERG)
        for iso in experiment_info:
            nuspec[i] += Nu_spec(labor).nu_rate_electronic(nu_comp[i], er_nu[i], iso, element)
        nu_rate[i] = np.trapz(nuspec[i], er_nu[i])
        print nu_comp[i], nu_rate[i]

    for i in range(len(identify)):
        er_nu_id[i] = np.logspace(np.log10(Qmin), np.log10(max_e_id[i]), NERG)
        for iso in experiment_info:
            nuspecLOOK[i] += Nu_spec(labor).nu_rate_electronic(identify[i], er_nu_id[i], iso, element)
        nu_rateLOOK[i] = np.trapz(nuspecLOOK[i], er_nu_id[i])
        print identify[i], nu_rateLOOK[i]

    add_chi = 0
    for i in range(len(identify)):
        if identify[i] == "reactor":
            nu_mean_f, nu_sig = reactor_flux(loc=labor)
        elif "geo" in identify[i]:
            nu_mean_f, nu_sig = geo_flux(loc=labor, el=identify[i][3:])
        else:
            nu_sig = NEUTRINO_SIG[identify[i]]
            nu_mean_f = NEUTRINO_MEANF[identify[i]]
        nu_sig *= red_uncer
        add_chi += (nu_mean_f / nu_sig)**2.

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

        nevts_NR = np.zeros(nu_contrib, dtype='int')
        nevts_n = np.zeros(nu_contrib, dtype='int')
        nevts_nLOOK = np.zeros(len(identify), dtype='int')
        nevts_bkg = np.zeros(len(bkg_list), dtype='int')

        fails = np.array([])
        nn = 0
        while nn < n_runs:
            print 'Run {:.0f} of {:.0f}'.format(nn + 1, n_runs)
            for i in range(nu_contrib):
                Nu_events_sim[i] = int(nu_rate[i] * MT)
                Nu_events_NR[i] = int(nu_rate_NR[i] * MT)
                try:
                    nevts_n[i] = poisson.rvs(int(Nu_events_sim[i]))
                except ValueError:
                    nevts_n[i] = 0
                try:
                    nevts_NR[i] = poisson.rvs(int(Nu_events_NR[i]))
                except ValueError:
                    nevts_NR[i] = 0

            for i in range(len(identify)):
                Nu_events_simLOOK[i] = int(nu_rateLOOK[i] * MT)
                try:
                    nevts_nLOOK[i] = poisson.rvs(int(Nu_events_simLOOK[i]))
                except ValueError:
                    nevts_nLOOK[i] = 0

            # Add electronic backgrounds...
            total_ev_bk = np.zeros(len(bkg_list))
            er_list_bk = np.zeros(len(bkg_list), dtype=object)
            spec_hold_bk = np.zeros(len(bkg_list), dtype=object)
            bkg_spec = np.zeros(len(bkg_list))
            for j,nme in enumerate(bkg_list):
                er_list_bk[j] = np.logspace(np.log10(Qmin), np.log10(Qmax), NERG)
                spec_hold_bk[j] = NEUTRINO_SPEC[nme](er_list_bk[j])
                total_ev_bk[j] = MT * np.trapz(spec_hold_bk[j], er_list_bk[j])
                nevts_bkg[j] = poisson.rvs(int(total_ev_bk[j]))

            if not QUIET:
                print 'Predicted Number of Nu BKG events: {}'.format(int(sum(Nu_events_sim)))
                print 'Predicted Number of Nu of interest events: {}'.format(Nu_events_simLOOK)
                print 'Predicted Number of Nu in Coherent Nuclear: {}'.format(Nu_events_NR)
                print 'Internal Bkg events: {}'.format(nevts_bkg)

            # Nuclear Error Reset
            NR_err = np.zeros_like(Nu_events_NR)
            uncertain_dict = {}
            for i in range(nu_contrib):
                
                if nu_comp[i] == "reactor":
                    nu_mean_f, nu_sig = reactor_flux(loc=labor)
                elif nu_comp[i] == "geoU":
                    nu_mean_f, nu_sig = geo_flux(loc=labor, el='U')
                elif nu_comp[i] == "geoTh":
                    nu_mean_f, nu_sig = geo_flux(loc=labor, el='Th')
                elif nu_comp[i] == "geoK":
                    nu_mean_f, nu_sig = geo_flux(loc=labor, el='K')
                else:
                    nu_mean_f = NEUTRINO_MEANF[nu_comp[i]]
                    nu_sig = NEUTRINO_SIG[nu_comp[i]]
                if Nu_events_NR[i] <= 10:
                    NR_err[i] = nu_sig
                else:
                    NR_err[i] = np.min([nu_sig, nu_mean_f / np.sqrt(Nu_events_NR[i])])
                uncertain_dict[nu_comp[i]] = NR_err[i]
                    
            if sim_files_exist:
                e_sim = np.array([])
                e_simLOOK = np.array([])
                e_simBKG = np.array([])
                for i in range(nu_contrib):
                    #print nu_comp[i], nevts_n[i]
                    u = random.rand(nevts_n[i]) * len(nu_events[i])
                    e_sim = np.append(e_sim, nu_events[i][u.astype(int)])
                for i in range(len(identify)):
                    #print identify[i], nevts_nLOOK[i]
                    u = random.rand(nevts_nLOOK[i]) * len(nu_eventsLOOK[i])
                    e_simLOOK = np.append(e_simLOOK, nu_eventsLOOK[i][u.astype(int)])
                for i in range(len(bkg_list)):
                    #print bkg_list[i], nevts_bkg[i]
                    u = random.rand(nevts_bkg[i]) * len(nuspecBKG[i])
                    e_simBKG = np.append(e_simBKG, nuspecBKG[i][u.astype(int)])
            else:
                print 'No Sim files... exiting...'
                exit()

            e_sim = np.concatenate((e_sim, e_simLOOK, e_simBKG))
            print 'Total Events: ', len(e_sim)

            # Simulate events
            print('Evaluated Events: Neutrino {:.0f}, NOI {:.0f}'.format(int(sum(nevts_n)), int(sum(nevts_nLOOK)) + int(sum(total_ev_bk))))


            if not QUIET:
                print 'Running Likelihood Analysis...'

            nu_bnds = [(-20.0, 3.0)] * nu_contrib
            full_bnds = [(-20.0, 3.0)] * (nu_contrib + len(identify) + len(bkg_list))

            like_init_tot = Likelihood_analysis('sigma_si', 'fnfp_si', 10., 0., 1.,
                                                MT, element, experiment_info, e_sim, np.zeros_like(e_sim),
                                                np.concatenate((nu_comp, identify, bkg_list)), labor,
                                                (nu_contrib+len(identify)+len(bkg_list)),
                                                np.concatenate((er_nu, er_nu_id, er_list_bk)),
                                                np.concatenate((nuspec, nuspecLOOK, spec_hold_bk)),
                                                np.concatenate((nu_rate, nu_rateLOOK, total_ev_bk)),
                                                Qmin, Qmax, reduce_uncer=red_uncer, DARK=False,
                                                uncertain_dict=uncertain_dict)

            skip_indx = [x for x in range(len(nu_comp), len(nu_comp) + len(identify) +len(bkg_list))]
            print skip_indx, nu_contrib, identify, bkg_list
            max_tot = minimize(like_init_tot.likelihood, np.zeros(nu_contrib + len(identify) + len(bkg_list)),
                               args=(np.array([-100.]), skip_indx),
                               tol=1e-4, method='SLSQP',
                               options={'maxiter': 100}, bounds=full_bnds, jac=like_init_tot.like_gradi)
#            print 'Max Total', max_tot.x, max_tot.fun

            def like_valuation(x, nu_contrib, skip_indx):
                
                nu_bnds = [(-20.0, 3.0)] * nu_contrib
                val = minimize(like_init_tot.like_multi_wrapper2, np.zeros(nu_contrib),
                         args=(x, np.array([-100.]), skip_indx),
                         tol=1e-4, method='SLSQP',
                         options={'maxiter': 100}, bounds=nu_bnds, jac=like_init_tot.like_multi_wrapper2_grad)
#                print 'Constrained Min: ', val.x, val.fun

                return val.fun

            # x = np.linspace(-3, 3, 50)
            # test = np.zeros_like(x)
            # for i in range(len(x)):
            #     test[i] = like_valuation(np.array([x[i]]), nu_contrib) - max_tot.fun
            # print np.column_stack((10.**x * nu_rateLOOK[0] * MT, test))
            # exit()

            # max_tot = minimize(like_init_tot.likelihood, np.zeros(nu_contrib + len(identify)),
            #                    args=(np.array([-100.]), range(nu_contrib, nu_contrib + len(identify))),
            #                    tol=1e-4, method='SLSQP',
            #                    options={'maxiter': 100}, bounds=full_bnds,
            #                    jac=like_init_tot.like_gradi)

            # print 'Minimizaiton Success: ', max_bkg.success, max_tot.success
            # print 'Values: ', max_bkg.fun, max_tot.fun
            #
            # if not max_bkg.success or not max_tot.success:
            #     fails = np.append(fails, nn)
            # test_stat = np.max([max_bkg.fun - max_tot.fun, 0.])
            set_zero = 10.**np.zeros(len(identify)) * -100.
            
            fix_to_zero = like_valuation(set_zero, nu_contrib + len(bkg_list), skip_indx)
            test_stat = np.max([fix_to_zero - max_tot.fun - add_chi, 0.])


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

            pval_arr = np.zeros_like(tstat_arr)
            for i in range(len(tstat_arr)):
                pval_arr[i] = chi2.sf(tstat_arr[i], len(identify))

            print 'Pvalue arr', pval_arr
            tot_entries = len(pval_arr)
            one_sigma = float(np.sum(pval_arr < 0.32)) / tot_entries
            two_sigma = float(np.sum(pval_arr < 0.05)) / tot_entries
            three_sigma = float(np.sum(pval_arr < 0.003)) / tot_entries

            print 'Sigmas: ', one_sigma, two_sigma, three_sigma

            print '~~~~~~~~~~~~~~~~~~~~~MOVING ON~~~~~~~~~~~~~~~~~~~~~'
            print '\n\n'
            if os.path.exists(file_info):
                load_old = np.loadtxt(file_info)
                new_arr = np.vstack((load_old, np.array([MT, one_sigma, two_sigma, three_sigma])))
                new_arr = new_arr[new_arr[:, 0].argsort()]
                np.savetxt(file_info, new_arr)
            else:
                np.savetxt(file_info, np.array([MT, one_sigma, two_sigma, three_sigma]))

    return



def identify_nu_naieve(exposure_low=1., exposure_high=100., expose_num=20, element='Germanium',
                       file_tag='', n_runs=20, Eth=-1, identify=np.array(['geoU', 'geoTh', 'geoK']),
                       red_uncer=1.):
    exposure_list = np.logspace(np.log10(exposure_low), np.log10(exposure_high), expose_num)

    print 'Run Info:'
    print 'Experiment: ', element
    print 'Identifying {} species: '.format(len(identify)), identify
    print '\n'

    # maxE = 0.
    # for i in identify:
    #     elem, Qmax, Qmin = Element_Info(element)
    #     maxER = Nu_spec(lab='Snolab').max_er_from_nu(NEUTRINO_EMAX[i], elem[0, 0])
    #     if maxER > maxE:
    #         maxE = maxER

    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenLAB)
    if Eth > 0:
        Qmin = Eth

    print 'Qmin: {:.2f}, Qmax: {:.2f}'.format(Qmin, Qmax)
    er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), 300)
    nu_comp = ['b8', 'b7l1', 'b7l2', 'pepl1', 'hep', 'pp', 'o15', 'n13', 'f17',
               'reactor', 'geoU', 'geoTh', 'geoK']

    keep_nus = []
    for i in range(len(nu_comp)):
        if nu_comp[i] in identify:
            if Nu_spec(labor).max_er_from_nu(NEUTRINO_EMAX[nu_comp[i]], experiment_info[0][0]) < Qmin:
                print 'Threshold too low for ', nu_comp[i]
                print 'Exiting...'
                exit()
            else:
                continue
        if Nu_spec(labor).max_er_from_nu(NEUTRINO_EMAX[nu_comp[i]], experiment_info[0][0]) > Qmin:
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
