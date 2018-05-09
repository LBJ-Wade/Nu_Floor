from rate_UV import *
from constants import *
from helpers import *
import numpy as np
from experiments import *
from scipy.interpolate import interp1d
import os

s_to_yr = 3.154*10.**7.
path = os.getcwd()

def make_bound(element='Xenon', model='sigma_si', Eth=-1.,
               exposure=1., eff='Perfect',
               mxrange=np.logspace(0., 3., 100), ngoal=3.18,
               time_info=False, GF=False, delta=0., fnfp=1., tag='_',
               guess=1e-45):
    
    
    experiment_info, Qmin, Qmax = Element_Info(element)
    if Eth > 0:
        Qmin = Eth
    #Qmax = Emax
    
    coupling = "fnfp" + model[5:]

    drdq_params = default_rate_parameters.copy()
    drdq_params['element'] = element
    drdq_params[coupling] = fnfp
    drdq_params['delta'] = delta
    drdq_params['GF'] = GF
    drdq_params['time_info'] = time_info

    if eff == 'Perfect':
        er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), 1000)
        efficiency = np.ones(len(er_list))
        file_eff = 'Idealized_'
    elif eff == 'LUX':
        er_list = np.logspace(np.log10(1.), np.log10(Qmax), 1000)
        luxeff = np.loadtxt(path + '/eff_Lux2016.dat')
        efficiency = interp1d(luxeff[:,0], luxeff[:,1], kind='cubic', bounds_error=False, fill_value=0.)(er_list)
        file_eff = 'LUX_'
        exposure = (3.35 * 10 ** 4. + 95.3 * 118.3) / (365.24 * 1000.)
        ngoal = 3.2
    elif eff == 'Xenon1T':
        #er_list = np.logspace(np.log10(1.), np.log10(Qmax), 1000)
        xeneff = np.loadtxt(path + '/Xenon1TC_Eff.dat')
        xeneff = xeneff[xeneff[:,0] <= 26.]
        er_list = np.logspace(np.log10(1.), np.log10(Qmax), 1000)
        #xeneff = np.loadtxt(path + '/Xenon1T_Eff.dat')
        #xeneff = xeneff[xeneff[:,0] <= 40.]
        efficiency = interp1d(xeneff[:,0], xeneff[:,1], kind='cubic', bounds_error=False, fill_value=0.)(er_list)
        file_eff = 'Xenon1T_'
        exposure = (2004. * 34.2) / (365.24 * 1000.)
        #exposure = (1042. * 34.2) / (365.24 * 1000.)
        ngoal = 2.7
    elif eff == 'PICO60':
        element = 'Fluorine'
        er_list = np.logspace(np.log10(1.), np.log10(Qmax), 1000)
        eff = np.loadtxt(os.getcwd() + '/pico60_eff.dat')
        efficiency = interp1d(eff[:,0],eff[:,1], kind='linear', bounds_error=False, fill_value=0.)(er_list)
        exposure = (1167.) / (365.24 * 1000.)
        file_eff = 'Pico60_'
        ngoal = 2.7
    elif eff == 'PandaX':
        er_list = np.logspace(np.log10(1.), np.log10(Qmax), 1000)
        xeneff = np.loadtxt(path + '/eff_Pandax2016C.dat')
        #xeneff = xeneff[xeneff[:,0] <= 50.]
        efficiency = interp1d(xeneff[:,0], xeneff[:,1], kind='linear', bounds_error=False, fill_value=0.)(er_list)
        file_eff = 'PandaX_'
        exposure = (5.4 * 10 ** 4.) / (365.24 * 1000.)
        #exposure = (77.1+79.6)*580. / (365.24 * 1000.)
        ngoal = 2.7
    elif eff == 'Darwin':
        er_list = np.logspace(np.log10(5.), np.log10(40.), 1000)
        efficiency = np.ones_like(er_list)
        file_eff = 'DARWIN_'
        exposure = (40000.*5.) / (1000.)
        ngoal = 2.7
    else:
        efficiency = np.zeros_like(er_list)
        file_eff = ''
        exit()


    for i,mass in enumerate(mxrange):
        mindm = np.zeros(len(experiment_info[:, 0]))
        for i, iso in enumerate(experiment_info):
            mindm[i] = MinDMMass(iso[0], delta, Qmin, vesc=533.+232.)
        MinDM = np.min(mindm)
        if mass < MinDM + 0.5:
            continue

        drdq_params[model] = guess
        drdq_params['mass'] = mass

        diff_r = dRdQ(er_list, np.zeros_like(er_list), **drdq_params)*10.**3.*s_to_yr * efficiency
        events = np.trapz(diff_r, er_list) * exposure
        sigma = np.log10(ngoal / (events / drdq_params[model]))
        if sigma < 0.:
            try:
                bounds = np.row_stack((bounds, np.array([mass, sigma])))
            except NameError:
                bounds = np.array([mass, sigma])

    filename = path + '/Bounds_Sve/'
    filename += file_eff + 'Bound_' + element + '_' + model + '_' + coupling + '_{:.2f}'.format(fnfp)
    filename += '_Exposure_{:.2f}_Eth_{:.2f}'.format(exposure, Qmin)
    filename += '_Ngoal_{:.2f}_delta_{:.2f}'.format(ngoal,delta) + tag + '.dat'

    np.savetxt(filename, bounds)
    return
