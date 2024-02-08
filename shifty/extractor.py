# -------------------------------------------------------------------------------------
# Third party imports
# -------------------------------------------------------------------------------------
import glob
import os
import sys

import numpy as np

import scipy
from scipy.spatial.distance import cdist as sp_dist
import sep
import scipy as sp

import warnings
warnings.filterwarnings("ignore")

#local imports
sys.path.append('/n/home05/malexandersen/Github/shifty')
from shifty import imagehandler

print(f"Starting extracting stars from images in {sys.argv[1]}", flush=True)
with open(sys.argv[1], 'r', encoding='utf-8') as list_file:
    list_lines = list_file.readlines()[:]
fits_files = []
for line in list_lines:
    fits_files.append(line.replace('\n',''))
print(len(fits_files), flush=True)
Reference = imagehandler.OneImage(fits_files[0], extno=0, EXPTIME='EXPOSURE', EXPUNIT='d', MAGZERO=18.,
                                  MJD_START='BJDREFI+TSTART+-2400000.5', GAIN='GAINA', FILTER='-Tess')
catalogs = {}
for fitf in fits_files:
    print(fitf, flush=True)
    num0 = fitf.split('_')[-1].split('.')[0]
    img_i = imagehandler.OneImage(fitf, extno=0, EXPTIME='EXPOSURE', EXPUNIT='d', MAGZERO=18.,
                                  MJD_START='BJDREFI+TSTART+-2400000.5', GAIN='GAINA', FILTER='-Tess')
    data = img_i.data.byteswap().newbyteorder()
    bkg = sep.Background(data)
    objects = sep.extract(data, 1.5, err=bkg.globalrms)
    #flux_aperture, flux_aperture_err, flux_aperture_flag = sep.sum_ellipse(data, objects['x'], objects['y'], objects['a'], objects['b'], objects['theta'], err=bkg.globalrms, gain=1.0)
    flux_aperture, flux_aperture_err, flux_aperture_flag = sep.sum_circle(data, objects['x'], objects['y'], 3, err=bkg.globalrms, gain=1.0)
    mjd = [img_i.header['SHIFTY_MJD_MID']]*len(flux_aperture)
    kd_tree = sp.spatial.KDTree(np.array([objects['x'], objects['y']]).T)
    catalog_i = {'flux_aperture':flux_aperture, 'flux_aperture_err':flux_aperture_err, 'flux_aperture_flag':flux_aperture_flag, 'mjd':mjd, 'kd_tree':kd_tree}
    for key in objects.dtype.names:
        catalog_i[key] = objects[key]
    catalogs[fitf] = catalog_i

clean_catalogs = {}
for fitf in fits_files[:]:
    cati = catalogs[fitf]
    num0 = fitf.split('_')[-1].split('.')[0]
    print(num0, flush=True)
    with open(fitf.replace('.fits','_transients.tsv'), 'w', encoding='utf-8') as write_file:
        with open(fitf.replace('.fits','_allsources.tsv'), 'w', encoding='utf-8') as all_file:
            write_file.write("#   1 RA                     Right Ascension                                            [deg]\n")  # 10
            write_file.write("#   2 DEC                    Declination                                                [deg]\n")
            write_file.write("#   3 X                      X pixel coordinate                                         [pix]\n")
            write_file.write("#   4 Y                      Y pixel coordinate                                         [pix]\n")
            write_file.write("#   5 FLUX                   Sum of member pixels in unconvolved data                   [count]\n")
            write_file.write("#   6 FLUX_APERTURE          Aperture flux  (3 pixel radius)                            [count]\n")
            write_file.write("#   7 FLUX_APERTURE_ERR      Error on FLUX_APERTURE                                     [count]\n")
            write_file.write("#   8 MJD                    Modified Julian Date at midpoint of observation            [days]\n")
            all_file.write("#   1 RA                     Right Ascension                                            [deg]\n")
            all_file.write("#   2 DEC                    Declination                                                [deg]\n")
            all_file.write("#   3 X                      X pixel coordinate                                         [pix]\n")  # 20
            all_file.write("#   4 Y                      Y pixel coordinate                                         [pix]\n")
            all_file.write("#   5 FLUX                   Sum of member pixels in unconvolved data                   [count]\n")
            all_file.write("#   6 FLUX_APERTURE          Aperture flux  (3 pixel radius)                            [count]\n")
            all_file.write("#   7 FLUX_APERTURE_ERR      Error on FLUX_APERTURE                                     [count]\n")
            all_file.write("#   8 MJD                    Modified Julian Date at midpoint of observation            [days]\n")
            n_near = np.zeros(len(cati['x']))
            for catk, catc in catalogs.items():
                numi = catk.split('_')[-1].split('.')[0]
                if (np.abs(int(numi)-int(num0))>10):
                    for i, idx in enumerate(cati['kd_tree'].query_ball_tree(catc['kd_tree'], 1)):  # 30
                        n_near[i] += len(idx)
                else:
                    continue
            for i, ni in enumerate(n_near):
                if (cati['a'][i]/cati['b'][i]<1.2) & (cati['flag'][i]==0) & (cati['flux_aperture_flag'][i]==0) & (not np.isnan(cati['flux_aperture'][i])):
                    radec = Reference.WCS.all_pix2world([[cati['x'][i], cati['y'][i]]], 0)[0]
                    all_file.write('\t'.join(list(radec.astype(str)) + [str(cati[k][i]) for k in ['x', 'y', 'flux', 'flux_aperture', 'flux_aperture_err', 'mjd']]) + '\n')
                    if (ni < 100):
                        write_file.write('\t'.join(list(radec.astype(str)) + [str(cati[k][i]) for k in ['x', 'y', 'flux', 'flux_aperture', 'flux_aperture_err', 'mjd']]) + '\n')
