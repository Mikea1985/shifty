'''
Tests of Classes / Methods for dealing with known objects.
'''


# -----------------------------------------------------------------------------
# Third party imports
# -----------------------------------------------------------------------------
import os
import sys
import numpy as np
import pytest
from pytest import mark

# -----------------------------------------------------------------------------
# Any local imports
# -----------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(
                os.path.realpath(__file__))))
from shifty import known, dev_tools as dev


# -----------------------------------------------------------------------------
# Constants and Test data
# -----------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
                        os.path.realpath(__file__))), 'dev_data')

all_times = np.arange(2458436.5, 2458464.5, 7)
# Some known RA and Dec values, for testing.
aeiOoME_sedna = [484.5211488, 0.8426141, 11.93068, 144.24763,  # a, e, i, O
                 311.35275, 358.11740, 2459000.5]  # o, M, Epoch

# -----------------------------------------------------------------------------
# Test "known" module
# -----------------------------------------------------------------------------


def test_empty_instantiation():
    '''
    Test the class instantiation with no input
    '''
    # Test creation of Known object
    K = known.Known()
    assert isinstance(K, known.Known), \
        'Object did not get created as expected'

    print('\t Completed test_empty_instantiation.')


names_of_variables = ('object_name', 'obs_code', 'times')
values_for_each_test = [
   ('Sedna', '568', all_times),
   ('101583', '568', all_times),
   ('Sedna', '000', all_times),
   ('101583', '000', all_times),
   ('Sedna', '500@-95', all_times),
   ('101583', '500@-95', all_times),
 ]
@pytest.mark.parametrize(names_of_variables, values_for_each_test)
def test__get_object_RADEC_from_horizons(object_name, obs_code, times):
    '''
    Test the private method for getting RA/Dec from Horizons for known objects.
    '''
    K = known.Known()
    K._get_object_RADEC_from_horizons(obs_code=obs_code, times=times,
                                      object_name=object_name)
    # Test that K.RA, K.Dec exist and have expected type.
    assert isinstance(K.RA, np.ndarray)
    assert isinstance(K.Dec, np.ndarray)
    # Test that K.RA, K.Dec have expected shape.
    assert np.shape(K.RA) == np.shape(times)
    assert np.shape(K.Dec) == np.shape(times)
    # Test that K.RA, K.Dec have expected values.
    expect_RADec = dev.radec_interp(times, dev.radec_from_file(object_name,
                                                               obs_code))
    assert np.all(np.isclose(K.RA, expect_RADec[0], atol=0.00000001, rtol=0))
    assert np.all(np.isclose(K.Dec, expect_RADec[1], atol=0.00000001, rtol=0))
    print('\t Completed test__get_object_RADEC_from_horizons.')


names_of_variables = ('object_name', 'obs_code', 'times')
values_for_each_test = [
   ('Sedna*', '568', all_times),
   ('101583*', '568', all_times),
   ('Sedna*', '000', all_times),
   ('101583*', '000', all_times),
   ('Sedna*', '500@-95', all_times),
   ('101583*', '500@-95', all_times),
 ]
@pytest.mark.parametrize(names_of_variables, values_for_each_test)
def test_get_known_RADEC_name(object_name, obs_code, times):
    '''
    Test the method for getting RA/Dec for a known object, using name.
    '''
    K = known.Known()
    K.get_known_RADEC(obs_code=obs_code, times=times, object_name=object_name)
    # Test that K.RA, K.Dec, K.times & K.obs_code exist and have expected type.
    assert isinstance(K.RA, np.ndarray)
    assert isinstance(K.Dec, np.ndarray)
    assert isinstance(K.times, type(times))
    assert isinstance(K.obs_code, str)
    # Test that K.RA, K.Dec, K.times have expected shape.
    assert np.shape(K.RA) == np.shape(times)
    assert np.shape(K.Dec) == np.shape(times)
    assert np.shape(K.times) == np.shape(times)
    # Test that K.RA, K.Dec, K.times and K.obs_code have expected values.
    expect_RADec = dev.radec_interp(times, dev.radec_from_file(object_name[:-1],
                                                               obs_code))
    assert np.all(np.isclose(K.RA, expect_RADec[0], atol=0.00000001, rtol=0))
    assert np.all(np.isclose(K.Dec, expect_RADec[1], atol=0.00000001, rtol=0))
    assert np.all(K.times == times)
    assert K.obs_code == obs_code
    print('\t Completed test_get_known_RADEC_name.')


names_of_variables = ('object_name', 'obs_code', 'times')
values_for_each_test = [
   ('Sedna*', '568', all_times),
   ('101583*', '568', all_times),
   ('Sedna*', '000', all_times),
   ('101583*', '000', all_times),
   ('Sedna*', '500@-95', all_times),
   ('101583*', '500@-95', all_times),
 ]
@pytest.mark.parametrize(names_of_variables, values_for_each_test)
def test_instantiate_with_object_name(object_name, obs_code, times):
    '''
    Test the class instantiation with an object name.
    '''
    K = known.Known(obs_code=obs_code, times=times, object_name=object_name)
    # Test that K.RA, K.Dec, K.times & K.obs_code exist and have expected type.
    assert isinstance(K.RA, np.ndarray)
    assert isinstance(K.Dec, np.ndarray)
    assert isinstance(K.times, type(times))
    assert isinstance(K.obs_code, str)
    # Test that K.RA, K.Dec, K.times have expected shape.
    assert np.shape(K.RA) == np.shape(times)
    assert np.shape(K.Dec) == np.shape(times)
    assert np.shape(K.times) == np.shape(times)
    # Test that K.RA, K.Dec, K.times and K.obs_code have expected values.
    expect_RADec = dev.radec_interp(times, dev.radec_from_file(object_name[:-1],
                                                               obs_code))
    assert np.all(np.isclose(K.RA, expect_RADec[0], atol=0.00000001, rtol=0))
    assert np.all(np.isclose(K.Dec, expect_RADec[1], atol=0.00000001, rtol=0))
    assert np.all(K.times == times)
    assert K.obs_code == obs_code
    print('\t Completed test_instantiate_with_object_name.')


names_of_variables = ('orbit', 'obs_code', 'times')
values_for_each_test = [
    pytest.param(aeiOoME_sedna, '568', all_times,
                 marks=mark.xfail(reason='Functionality not implemented.')),
 ]
@pytest.mark.parametrize(names_of_variables, values_for_each_test)
def test__get_orbit_RADEC(orbit, obs_code, times):
    '''
    Test the private method for getting RA/Dec from an orbit.
    '''
    K = known.Known()
    K._get_orbit_RADEC(obs_code=obs_code, times=times, orbit=orbit)
    # Test that K.RA, K.Dec exist and have expected type.
    assert isinstance(K.RA, np.ndarray)
    assert isinstance(K.Dec, np.ndarray)
    # Test that K.RA, K.Dec have expected shape.
    assert np.shape(K.RA) == np.shape(times)
    assert np.shape(K.Dec) == np.shape(times)
    # Test that K.RA, K.Dec have expected values.
    expect_RADec = dev.radec_interp(times, dev.radec_from_file('Sedna',
                                                               obs_code))
    assert np.all(np.isclose(K.RA, expect_RADec[0], atol=0.00000001, rtol=0))
    assert np.all(np.isclose(K.Dec, expect_RADec[1], atol=0.00000001, rtol=0))
    print('\t Completed test__get_orbit_RADEC.')


names_of_variables = ('orbit', 'obs_code', 'times')
values_for_each_test = [
    pytest.param(aeiOoME_sedna, '568', all_times,
                 marks=mark.xfail(reason='Functionality not implemented.')),
 ]
@pytest.mark.parametrize(names_of_variables, values_for_each_test)
def test_get_known_RADEC_orbit(orbit, obs_code, times):
    '''
    Test the method for getting RA/Dec for a known object, using orbit.
    '''
    K = known.Known()
    K.get_known_RADEC(obs_code=obs_code, times=times, orbit=orbit)
    # Test that K.RA, K.Dec, K.times & K.obs_code exist and have expected type.
    assert isinstance(K.RA, np.ndarray)
    assert isinstance(K.Dec, np.ndarray)
    assert isinstance(K.times, type(times))
    assert isinstance(K.obs_code, str)
    # Test that K.RA, K.Dec, K.times have expected shape.
    assert np.shape(K.RA) == np.shape(times)
    assert np.shape(K.Dec) == np.shape(times)
    assert np.shape(K.times) == np.shape(times)
    # Test that K.RA, K.Dec, K.times and K.obs_code have expected values.
    expect_RADec = dev.radec_interp(times, dev.radec_from_file('Sedna',
                                                               obs_code))
    assert np.all(np.isclose(K.RA, expect_RADec[0], atol=0.00000001, rtol=0))
    assert np.all(np.isclose(K.Dec, expect_RADec[1], atol=0.00000001, rtol=0))
    assert np.all(K.times == times)
    assert K.obs_code == obs_code
    print('\t Completed test_get_known_RADEC_orbit.')


names_of_variables = ('object_name', 'times')
values_for_each_test = [
    ('Sedna', all_times),
 ]
@pytest.mark.parametrize(names_of_variables, values_for_each_test)
def test__get_object_XYZ_from_horizons(object_name, times):
    '''
    Test the private method for getting XYZ for a known object from Horizons.
    '''
    K = known.Known()
    K._get_object_XYZ_from_horizons(times=times, object_name=object_name)
    # Test that K.XYZ exist and have expected type.
    assert isinstance(K.XYZ, np.ndarray)
    # Test that K.XYZ have expected shape.
    assert np.shape(K.XYZ) == (3, np.shape(times)[0])
    # Test that K.XYZ have expected values.
    expect_XYZ = dev.xyz_interp(times, dev.xyz_from_file('Sedna'))
    #0.000000000001 AU is 15 cm
    assert np.all(np.isclose(K.XYZ, expect_XYZ, atol=0.000000000001, rtol=0))
    print('\t Completed test__get_object_XYZ_from_horizons.')


# -------------------------------------------------------------------------
# Test data & convenience functions
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# If this is run as main, run all the tests.
# -------------------------------------------------------------------------
# Won't need these calls if use pytest/similar
if __name__ == '__main__':
    test_empty_instantiation()
    test__get_object_RADEC_from_horizons('Sedna', '568', all_times)
    test_get_known_RADEC_name('Sedna', '568', all_times)
    test_instantiate_with_object_name('Sedna', '568', all_times)
#    test__get_orbit_RADEC(orbit, '568', all_times)
#    test_get_known_RADEC_orbit(orbit, '568' all_times)
    test__get_object_XYZ_from_horizons('Sedna', all_times)

# End of file
