from rate_UV import *
from constants import *
from helpers import *
import numpy as np
from experiments import *
from scipy.interpolate import interp1d
import os

s_to_yr = 3.154*10.**7.
path = os.getcwd()

def make_bound(element='Xenon', model='sigma_si', Eth=0.1,
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
    filename += '_Exposure_{:.2f}_Eth_{:.2f}'.format(exposure, Eth)
    filename += '_Ngoal_{:.2f}'.format(ngoal) + tag + '.dat'

    np.savetxt(filename, bounds)
    return
