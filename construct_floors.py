import numpy as np
import glob
import os
from scipy.interpolate import interp1d
from scipy.optimize import brentq

path = os.getcwd()


def make_a_floor(element='germanium', model='sigma_si', fnfp=1., exposure=1.,
                 delta=0., tag='_', qaim=9.):

    coupling = "fnfp" + model[5:]
    file_info = path + '/Saved_Files/'
    file_info += element + '_' + model + '_' + coupling + '_{:.2f}'.format(fnfp)
    file_info += '_Exposure_{:.2f}_tonyr_DM_Mass_*_GeV'.format(exposure)
    file_info += tag + '.dat'

    file_sv = path + '/Floors/' + element + '_' + model + '_' + coupling + '_{:.2f}'.format(fnfp)
    file_sv += '_Exposure_{:.2f}_tonyr_QGoal_{:.2f}'.format(exposure, qaim)
    file_sv += tag + '.dat'

    print 'Look for files of form: ', file_info
    files = glob.glob(file_info)
    print files
    for f in files:
        mx = float(f[f.find('DM_Mass_')+8:f.find('_GeV')])
        load = np.loadtxt(f)
        try:
            dim_test = load.shape[1]
            csec = brentq(lambda x: interpola(x, load[:,0], load[:, 1]) - qaim, load[0, 0], -40.)
            print 'DM mass: {:.2f}, Cross Sec {:.2e}'.format(mx, 10.**csec)

            if os.path.exists(file_sv):
                load_old = np.loadtxt(file_sv)
                if mx not in load_old[:, 0]:
                    new_arr = np.vstack((load_old, np.array([mx, csec])))
                    new_arr = new_arr[new_arr[:, 0].argsort()]
                    np.savetxt(file_sv, new_arr)
                else:
                    print 'DM mass already in file...'
            else:
                np.savetxt(file_sv, np.array([mx, csec]))

        except IndexError:
            pass
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