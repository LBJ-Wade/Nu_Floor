import numpy as np
import glob
import os
from scipy.interpolate import interp1d
from scipy.optimize import brentq, curve_fit
from scipy.stats import gaussian_kde as kde
from statsmodels.nonparametric.smoothers_lowess import lowess
from helpers import gauss_cdf_function
from experiments import *

path = os.getcwd()


def make_a_floor(element='Germanium', model='sigma_si', fnfp=1., exposure=1.,
                 delta=0., tag='_', qaim=0.9, eth=0.1, xen='LZ', lab=''):

    if len(lab) == 0:
        lab = laboratory(element, xen=xen)

    coupling = "fnfp" + model[5:]
    file_info = path + '/Saved_Files/'
    file_info += element + '_' + model + '_' + coupling + '_{:.2f}'.format(fnfp)
    file_info += '_Exposure_{:.2f}_tonyr_DM_Mass_*_GeV'.format(exposure)
    file_info += '_Eth_{:.2f}_'.format(eth) + lab + '_' + tag + '.dat'

    file_sv = path + '/Floors/' + element + '_' + model + '_' + coupling + '_{:.2f}'.format(fnfp)
    file_sv += '_Exposure_{:.2f}_tonyr_QGoal_{:.2f}'.format(exposure, qaim)
    file_sv += tag + '.dat'

    try:
        os.remove(file_sv)
    except:
        pass

    print 'Look for files of form: ', file_info
    files = glob.glob(file_info)
    #print files
    for f in files:
        mx = float(f[f.find('DM_Mass_')+8:f.find('_GeV')])
        load = np.loadtxt(f)
        try:
            print 'DM mass: {:.2f}'.format(mx)
            dim_test = load.shape[1]
            rm_ind = [idx for idx, item in enumerate(load[:,0]) if item in load[:,0][:idx]]
            useable = np.delete(load, rm_ind, axis=0)
            try:
                mean = sum(useable[:, 0] * useable[:, 1]) / sum(useable[:, 1])
                popt, pcov = curve_fit(gauss_cdf_function, useable[:, 0], useable[:, 1], p0=[mean, 1.])
                csec = brentq(lambda x: gauss_cdf_function(x, *popt) - qaim, -60., -30.)
                print 'DM mass: {:.2f}, Cross Sec {:.2e}'.format(mx, 10. ** csec)
            except ValueError:
                continue

            if os.path.exists(file_sv):
                load_old = np.loadtxt(file_sv)

                if mx not in load_old:
                    new_arr = np.vstack((load_old, np.array([np.log10(mx), csec])))
                    new_arr = new_arr[new_arr[:, 0].argsort()]
                    np.savetxt(file_sv, new_arr)
                else:
                    print 'DM mass already in file...'
            else:
                np.savetxt(file_sv, np.array([np.log10(mx), csec]))

        except IndexError:
            pass

    # try:
    #     load = np.loadtxt(file_sv)
    #     if len(load) > 3:
    #         new_arr = lowess(load[:,1], load[:,0], frac=0.2, return_sorted=True)
    #         np.savetxt(file_sv, new_arr)
    # except IOError:
    #     print 'No Files Found.'
    return


def interpola(val, x, y):
    try:
        f = np.zeros(len(val))
        for i, v in enumerate(val):
            if v <= x[0]:
                f[i] = y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (v - x[0])
            elif v >= x[-1]:
                f[i] = y[-2] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (v - x[-2])
            else:
                f[i] = interp1d(x, y, kind='cubic').__call__(v)
    except TypeError:
        if val <= x[0]:
            f = y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (val - x[0])
        elif val >= x[-1]:
            f = y[-2] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (val - x[-2])
        else:
            try:
                f = interp1d(x, y, kind='cubic').__call__(val)
            except:
                f = interp1d(x, y, kind='linear').__call__(val)
    return f