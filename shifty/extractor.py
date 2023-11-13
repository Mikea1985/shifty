fits_files = sorted(glob.glob('bright_ansep_b3.0f1.0h40d10_???.fits'))[:]
Reference = imagehandler.OneImage(fits_files[0], extno=0, EXPTIME='EXPOSURE', EXPUNIT='d', MAGZERO=18.,
                                  MJD_START='BJDREFI+TSTART+-2400000.5', GAIN='GAINA', FILTER='-Tess')
catalogs = {}
for fitf in fits_files:
    print(fitf)
    num0 = fitf.split('_')[-1].split('.')[0]
    img_i = imagehandler.OneImage(fitf, extno=0, EXPTIME='EXPOSURE', EXPUNIT='d', MAGZERO=18.,
                                  MJD_START='BJDREFI+TSTART+-2400000.5', GAIN='GAINA', FILTER='-Tess')
    data = img_i.data.byteswap().newbyteorder()
    bkg = sep.Background(data)
    objects = sep.extract(data, 1.5, err=bkg.globalrms)
    #flux_aperture, flux_aperture_err, flux_aperture_flag = sep.sum_ellipse(data, objects['x'], objects['y'], objects['a'], objects['b'], objects['theta'], err=bkg.globalrms, gain=1.0)
    flux_aperture, flux_aperture_err, flux_aperture_flag = sep.sum_circle(data, objects['x'], objects['y'], 3, err=bkg.globalrms, gain=1.0)
    mjd = [img_i.header['SHIFTY_MJD_MID']]*len(flux_aperture)
    catalog_i = {'flux_aperture':flux_aperture, 'flux_aperture_err':flux_aperture_err, 'flux_aperture_flag':flux_aperture_flag, 'mjd':mjd}
    for key in objects.dtype.names:
        catalog_i[key] = objects[key]
    catalogs[fitf] = catalog_i

clean_catalogs = {}
for fitf in fits_files:
    cati = catalogs[fitf]
    num0 = fitf.split('_')[-1].split('.')[0]
    print(num0)
    with open(fitf.replace('.fits','_transients.tsv'), 'w', encoding='utf-8') as write_file:
        write_file.write("#   1 RA                     Right Ascension                                            [deg]\n")
        write_file.write("#   2 DEC                    Declination                                                [deg]\n")
        write_file.write("#   3 X                      X pixel coordinate                                         [pix]\n")
        write_file.write("#   4 Y                      Y pixel coordinate                                         [pix]\n")
        write_file.write("#   5 FLUX                   Sum of member pixels in unconvolved data                   [count]\n")
        write_file.write("#   6 FLUX_APERTURE          Aperture flux  (3 pixel radius)                            [count]\n")
        write_file.write("#   7 FLUX_APERTURE_ERR      Error on FLUX_APERTURE                                     [count]\n")
        write_file.write("#   8 MJD                    Modified Julian Date at midpoint of observation            [days]\n")
        for i in np.arange(len(cati['x'])):
            n_near = 0
            for catk, catc in catalogs.items():
                numi = catk.split('_')[-1].split('.')[0]
                if (np.abs(int(numi)-int(num0))>100):
                    n = np.sum(sp_dist(np.array([catc['x'], catc['y']]).T, [[cati['x'][i], cati['y'][i]]])<1)
                    n_near += n
                else:
                    continue
            #print(num0, (n_near < 100) , (cati['a'][i]/cati['b'][i]<1.2) , (cati['flag'][i]==0) , (cati['flux_aperture_flag'][i]==0))
            if (n_near < 100) & (cati['a'][i]/cati['b'][i]<1.2) & (cati['flag'][i]==0) & (cati['flux_aperture_flag'][i]==0):
            #if (cati['a'][i]/cati['b'][i]<1.2):
                radec = Reference.WCS.all_pix2world([[cati['x'][i], cati['y'][i]]], 0)[0]
                write_file.write('\t'.join(list(radec.astype(str)) + [str(cati[k][i]) for k in ['x', 'y', 'flux', 'flux_aperture', 'flux_aperture_err', 'mjd']]) + '\n')

