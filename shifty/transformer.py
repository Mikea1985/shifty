# -*- coding: utf-8 -*-
# shifty/shifty/transformer.py

'''
Classes / methods for transforming to/from abg.
Provides methods to
-
'''

# -----------------------------------------------------------------------------
# Third party imports
# -----------------------------------------------------------------------------
import os
import sys
import getpass
# from datetime import datetime
# import copy
import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.time import Time

# -----------------------------------------------------------------------------
# Any local imports
# -----------------------------------------------------------------------------
# Different machines set up differently ...
# ... adding paths to force stuff to work while developing
if getpass.getuser() in ['matthewjohnpayne']:
    sys.path.append('/Users/matthewjohnpayne/Envs/mpcvenv/')
else:
    pass

sys.path.append(os.path.dirname(os.path.dirname(
                os.path.realpath(__file__))))
# from shifty.known import Known

# -----------------------------------------------------------------------------
# Define some constants
# -----------------------------------------------------------------------------

# ECL = np.radians(23.43928)
ECL = obliquityJ2000 = (84381.448 * (1. / 3600) * np.pi / 180.)  # Obliquity of ecliptic at J2000

# -----------------------------------------------------------------------------
# Various class definitions for *data import * in shifty
# -----------------------------------------------------------------------------

# Turning off some stupid syntax-checker warnings:
# pylint: disable=too-many-instance-attributes, too-few-public-methods, too-many-arguments, attribute-defined-outside-init, no-member, import-outside-toplevel


class Transformer():
    '''
    Class for dealing with an object provided only tangent-plane parameters,
    alpha, beta, gamma, alpha-dot, beta-dot, gamma-dot (abg for short).
    '''

    def __init__(self, times=None, obs_code=None, method='MPC', verbose=False):
        '''
        When object is initialized, either initialize blank object,
        or initialize with an array of times and an obs_code.
        When initialized with times & obs_code, calculate observer position
        right away. This way, for a given set of images, the same object can
        be used to calculate thetas from several different abg arrays, saving
        computation as the observer location only has to be computed once.
        '''
        self.times = times
        self.obs_code = obs_code
        self.method = method
        self.verbose = verbose
        self.abg = None
        self.time0 = None
        # if instantiated with times and obs_code, pre-calculate observer position at all times
        # in heliocentric eclipctic space. This pre-calculation thus only needs to be done once
        # even if multiple different time0 or abg's want to be used.
        if times is not None and obs_code:
            self.observer_helio_ecliptic_xyz = self._observer_heliocentric_ecliptic_XYZ(self.times)
        else:
            self.observer_helio_ecliptic_xyz = None

    def __call__(self, abg, time0, latlon0, wcs=None, verbose=None):
        '''
        Calculate thetas for a given set of ABG, reference time and reference
        latitude & longitude.
        If a WCS is also given, convert the thetas to pixel shifts.
        '''
        if verbose:  # Change self.verbose if verbose keyword defined.
            self.verbose = verbose
        self.abg = abg
        self.time0 = time0
        self.latlon0 = latlon0
        if wcs is not None:
            # Calculate pixel shifts from thetas
            thetas = self.abg2theta()
            pixels = self.thetas2pix(thetas, wcs)
            return pixels.T - np.min(pixels, 1)
        # else:
        return self.abg2theta()

    def abg2theta(self, GM=c.GM_sun.to('au**3/year**2').value):
        '''
        Converts input abg to a theta vector at time dtime from reference time.
        inputs:
        -------
        abg    - array length 6 - array containing alpha, beta, gamma,
                                  alpha-dot, beta-dot, gamma-dot.
        timesJD - float - times in Julian days
        '''
        # convert times to YEARS since reference time,
        # accounting for light travel time
        light_travel_time = self._get_light_travel_times()
        dtime = (self.times - self.time0) * u.day.to(u.yr) - light_travel_time
        # Calculate gravitational effect
        grav_x, grav_y, grav_z = self._get_gravity_vector(dtime, GM)
        # XYZ of observer:
        # flake8: W503
        x_E, y_E, z_E = self.get_observer_xyz_projected().T  # pylint: disable=unpacking-non-sequence
        num_x = (self.abg[0] + self.abg[3] * dtime
                 + self.abg[2] * grav_x - self.abg[2] * x_E)
        num_y = (self.abg[1] + self.abg[4] * dtime
                 + self.abg[2] * grav_y - self.abg[2] * y_E)
        denominator = (1 + self.abg[5] * dtime
                       + self.abg[2] * grav_z - self.abg[2] * z_E)
        theta_x = num_x / denominator                       # eq 6
        theta_y = num_y / denominator                       # eq 6
        # theta_x = abg[0] + abg[3] * dtime - abg[2] * x_E   # eq 16
        # theta_y = abg[1] + abg[4] * dtime - abg[2] * y_E   # eq 16
        if self.verbose:
            print("Thetas are in radians!!!")
        return np.array([theta_x, theta_y]).T  # These are radians!

    def _get_light_travel_times(self):
        '''
        Calculate the light travel time from the object to the observer.
        Units are years.
        '''
        # For now, just simplify and assume distance = 1/gamma
        ltts = (1 / self.abg[2] * (u.au / c.c).to(u.yr)).value
        return ltts

    def _get_gravity_vector(self, dtime, GM=c.GM_sun.to('au**3/year**2').value):
        '''
        g(t), the gravitational perturbation vector, calculated from equations
        (2), (3) and (4) of B&K 2000.
        For now, just using perturbations from sun only, sufficient for TNOs & t<<1 yr.
        '''
        acc_z = - GM * self.abg[2] ** 2
        grav_x, grav_y, grav_z = 0, 0, 0.5 * acc_z * dtime ** 2
        return grav_x, grav_y, grav_z

    def get_observer_xyz_projected(self):
        '''
        X_E(t) vector.
        Calculates the locations of the observer relative to the reference,
        in projection coordinate frame.
        '''
        # Observer's heliocentric ecliptic location at all times.
        if self.observer_helio_ecliptic_xyz is not None:
            observer_helio_ecliptic = self.observer_helio_ecliptic_xyz
        else:
            observer_helio_ecliptic = self._observer_heliocentric_ecliptic_XYZ(self.times)
        # Observer's heliocentric ecliptic location at reference times.
        observer_helio_ecliptic0 = self._observer_heliocentric_ecliptic_XYZ(self.time0)
        # Observer's ecliptic location relative to the location at the reference time
        observer_helio_ecliptic_relative = observer_helio_ecliptic - observer_helio_ecliptic0
        # Convert observer location to projection coordinate system.
        observer_projected = np.array([xyz_ec_to_proj(*obspos, *self.latlon0)
                                       for obspos in observer_helio_ecliptic_relative])
        return observer_projected

    def _observer_heliocentric_ecliptic_XYZ(self, times):
        '''
        Get the heliocentric ecliptic position of the observer.
        If reference=True, use reference time, otherwise all times.
        '''
        args = {'times': times, 'obs_code': self.obs_code,
                'verbose': self.verbose}
        # Use Horizons if explicitly requested (or if stupid JPL obs_code):
        if (len(self.obs_code) != 3) | (self.method == 'JPL'):
            return _observer_heliocentric_ecliptic_XYZ_from_JPL(**args)
        # Otherwise, try using MPC tools first.
        try:
            return _observer_heliocentric_ecliptic_XYZ_from_MPC(**args)
        except (ModuleNotFoundError, ValueError, NameError):
            # If MPC tools fail, use Horizons anyway
            return _observer_heliocentric_ecliptic_XYZ_from_JPL(**args)

    def thetas2pix(self, thetas, wcs):
        '''
        Use a given WCS to convert thetas (in arc-seconds) to pixel shift.
        '''
        # Convert thetas to longitudes and latitudes
        latlon_from_abg = proj_to_ec(thetas[:, 0], thetas[:, 1], *self.latlon0)
        # Convert latitutde and longitude to RA and Dec
        radec_from_abg = np.degrees(ec_to_eq(*latlon_from_abg))
        # Convert RA and Dec to pixel coordinates
        if np.shape(wcs) == ():
            pix_from_abg = np.array(wcs.all_world2pix(radec_from_abg[0, :],
                                                      radec_from_abg[1, :],
                                                      0))[::-1]
        else:  # assume one wcs per image
            pix_list = []
            for i, wcsi in enumerate(wcs):
                pix_list.append(wcsi.all_world2pix(radec_from_abg[0, i],
                                                   radec_from_abg[1, i],
                                                   0))
            pix_from_abg = np.array(pix_list).T[::-1]
        return pix_from_abg


def _observer_heliocentric_ecliptic_XYZ_from_JPL(times, obs_code='500',
                                                 verbose=False):
    '''
    Query horizons for the ECLIPTIC heliocentric
    observatory position at a sequence of times.

    input:
    obs_code    - string
                - Note that Horizons uses some weird ones sometimes,
                  like "500@-95" for Tess.
    times       - array of JD times (UTC)
    '''
    # Import here rather than at top level, to avoid loading if unneccessary.
    from astroquery.jplhorizons import Horizons
    times_AP = Time(times, format='jd', scale='utc')
    # convert times to tdb, the time system used by Horizons for vectors.
    times_tdb = times_AP.tdb.value
    horizons_query = Horizons(id='10', location=obs_code,
                              epochs=times_tdb, id_type='id')
    horizons_vector = horizons_query.vectors(refplane='ecliptic')
    helio_OBS_ecl = 0 - np.array([horizons_vector['x'], horizons_vector['y'],
                                  horizons_vector['z']]).T
    if verbose:
        print('No verbosity implemented yet, sorry')
    return helio_OBS_ecl


def _observer_heliocentric_ecliptic_XYZ_from_MPC(times, obs_code='500',
                                                 verbose=False):
    '''
    Get the heliocentric ECLIPTIC vector coordinates of the
    observatory at the time jd_utc.

    input:
    obs_code    - string
    times       - JD time (UTC)
    '''
    helio_OBS_equ = _observer_heliocentric_equatorial_XYZ_from_MPC(times,
                                                                   obs_code,
                                                                   verbose)
    helio_OBS_ecl = []
    for hequ in helio_OBS_equ:
        helio_OBS_ecl.append(_equatorial_to_ecliptic(hequ))
    return np.array(helio_OBS_ecl)


def _observer_heliocentric_equatorial_XYZ_from_MPC(times, obs_code='500',
                                                   verbose=False):
    '''
    Get the heliocentric EQUATORIAL vector coordinates of the
    observatory at the time jd_utc.
    '''
    # MPC_library imported here, as it is an optional dependancy
    from mpcpp import MPC_library as mpc
    obsCodes = mpc.Observatory()
    helio_OBS_equ = []
    # Make sure time is an array or list
    times_utc = np.array([times]) if (isinstance(times, int) |
                                      isinstance(times, float)) else times
    for jd_utc in times_utc:
        hom = obsCodes.getObservatoryPosition(obsCode=obs_code, jd_utc=jd_utc,
                                              old=False)
        helio_OBS_equ.append(hom)
        if verbose:
            print('MPC XYZ:')
            print(f'Heliocentric position of observatory: {hom} au\n')

    return np.array(helio_OBS_equ)


def _equatorial_to_ecliptic(input_xyz):
    '''
    Convert an cartesian vector from mean equatorial to mean ecliptic.
    input:
        input_xyz              - np.array length 3
    output:
        output_xyz - np.array length 3
    '''
    # MPC_library imported here, as it is an optional dependancy
    rotation_matrix = get_rotation_matrix()
    output_xyz = np.dot(rotation_matrix, input_xyz.reshape(-1, 1)).flatten()
    return output_xyz


def get_rotation_matrix():
    '''
    This function is inspired by the "rotate_matrix" function in the
    MPC_library, but is placed here to reduce non-trivial dependencies.
    '''
    cose = np.cos(obliquityJ2000)
    sine = np.sin(-obliquityJ2000)
    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, cose, sine],
                                [0.0, -sine, cose]])
    return rotation_matrix


# -------------------------------------------------------------------------
# These functions really don't need to be methods, and therefore aren't.
# They are more versatile (and easier to test) as functions.
# No need to over-complicate things.
# -------------------------------------------------------------------------

'''Here are some functions piped from Bernstein's transform.c, part of orbfit'''


def check_latlon0(lat0, lon0):
    '''
    This function is adapted from Bernstein's transform.c, part of orbfit.
    It doesn't actually check anything here,
    just calculates all the sines and cosines.
    '''
    clat0 = np.cos(lat0)
    slat0 = np.sin(lat0)
    clon0 = np.cos(lon0)
    slon0 = np.sin(lon0)

    return clat0, slat0, clon0, slon0


def ec_to_proj(lat_ec, lon_ec, lat0, lon0):
    '''
    First routine goes from ecliptic lat/lon to projected x/y angles

    This function is adapted from Bernstein's transform.c, part of orbfit.
    '''
    clat0, slat0, _, _ = check_latlon0(lat0, lon0)
    cdlon = np.cos(lon_ec - lon0)
    sdlon = np.sin(lon_ec - lon0)
    clat = np.cos(lat_ec)
    slat = np.sin(lat_ec)

    xp = clat * sdlon
    yp = clat0 * slat - slat0 * clat * cdlon
    zp = slat0 * slat + clat0 * clat * cdlon

    return xp / zp, yp / zp


def proj_to_ec(x_proj, y_proj, lat0, lon0):
    '''
    Now the inverse, from projected xy to ecliptic lat/lon

    This function is adapted from Bernstein's transform.c, part of orbfit.
    '''
    clat0, slat0, _, _ = check_latlon0(lat0, lon0)

    zp = 1. / np.sqrt(1 + x_proj * x_proj + y_proj * y_proj)
    lat_ec = np.arcsin(zp * (slat0 + y_proj * clat0))
    lon_ec = lon0 + np.arcsin(x_proj * zp / np.cos(lat_ec))

    return lat_ec, lon_ec


def xyz_ec_to_proj(x_ec, y_ec, z_ec, lat0, lon0):
    '''
    Next go from x,y,z in ecliptic orientation to x,y,z in tangent-point orientiation.

    This function is adapted from Bernstein's transform.c, part of orbfit.
    '''
    clat0, slat0, clon0, slon0 = check_latlon0(lat0, lon0)

    x_p = -slon0 * x_ec + clon0 * y_ec
    y_p = -clon0 * slat0 * x_ec - slon0 * slat0 * y_ec + clat0 * z_ec
    z_p = clon0 * clat0 * x_ec + slon0 * clat0 * y_ec + slat0 * z_ec

    return x_p, y_p, z_p


def xyz_proj_to_ec(x_p, y_p, z_p, lat0, lon0):
    '''
    And finally from tangent x,y,z to ecliptic x,y,z.

    This function is adapted from Bernstein's transform.c, part of orbfit.
    '''
    clat0, slat0, clon0, slon0 = check_latlon0(lat0, lon0)

    x_ec = -slon0 * x_p - clon0 * slat0 * y_p + clon0 * clat0 * z_p
    y_ec = clon0 * x_p - slon0 * slat0 * y_p + slon0 * clat0 * z_p
    z_ec = clat0 * y_p + slat0 * z_p

    return x_ec, y_ec, z_ec


def eq_to_ec(ra_eq, dec_eq):
    '''
    First takes RA,DEC in equatorial to ecliptic.

    This function is adapted from Bernstein's transform.c, part of orbfit.
    '''
    se = np.sin(ECL)
    ce = np.cos(ECL)
    sd = ce * np.sin(dec_eq) - se * np.cos(dec_eq) * np.sin(ra_eq)
    lat_ec = np.arcsin(sd)

    y = ce * np.cos(dec_eq) * np.sin(ra_eq) + se * np.sin(dec_eq)
    x = np.cos(dec_eq) * np.cos(ra_eq)
    lon_ec = np.arctan2(y, x)

    return lat_ec, lon_ec


def xyz_eq_to_ec(x_eq, y_eq, z_eq):
    '''
    And transform x,y,z from eq to ecliptic

    This function is adapted from Bernstein's transform.c, part of orbfit.
    '''
    se = np.sin(ECL)
    ce = np.cos(ECL)

    x_ec = x_eq
    y_ec = ce * y_eq + se * z_eq
    z_ec = -se * y_eq + ce * z_eq

    return x_ec, y_ec, z_ec


def ec_to_eq(lat_ec, lon_ec):
    '''
    And transform x,y,z from eq to ecliptic.

    To reverse above, just flip sign of ECL effectively.
    '''
    se = np.sin(-ECL)
    ce = np.cos(ECL)

    sd = ce * np.sin(lat_ec) - se * np.cos(lat_ec) * np.sin(lon_ec)
    dec_eq = np.arcsin(sd)

    y = ce * np.cos(lat_ec) * np.sin(lon_ec) + se * np.sin(lat_ec)
    x = np.cos(lat_ec) * np.cos(lon_ec)
    ra_eq = np.arctan2(y, x)

    return ra_eq, dec_eq


def xyz_ec_to_eq(x_ec, y_ec, z_ec):
    '''
    And transform x,y,z from ecliptic to eq.

    To reverse above, just flip sign of ECL effectively.
    '''
    se = np.sin(-ECL)
    ce = np.cos(ECL)

    x_eq = x_ec
    y_eq = ce * y_ec + se * z_ec
    z_eq = -se * y_ec + ce * z_ec

    return x_eq, y_eq, z_eq

# END
