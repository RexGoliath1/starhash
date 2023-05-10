# This script generates a CSV file containing the Hipparcos catalog
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord, Distance
import astropy.units as u
from astropy.time import Time
import os
import numpy as np


# set the reference time for proper motion calculations
ref_time = Time('2000-01-01')

# set the date to calculate the updated positions (Use now for testing)
update_time = Time.now()

catalog = "I/311/hip2"
output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hipparcos.csv")

columns = [
    "RAJ2000",
    "DEJ2000",
    "HIP",
    "RArad",
    "DErad",
    "Plx",
    "pmRA",
    "pmDE",
    "Hpmag",
    "B-V"
]

ra = 0.0
dec = 0.0

v = Vizier(catalog=catalog, columns=columns, row_limit=-1)

result = v.query_constraints(Hpmag='>0.0')
if result is None:
    raise ValueError('No results found')
else:
    result = result[0]

    # There are apparently some negative parallaxes in the Hipparcos and Gaia catalogs. 
    # These are converted to NaN distances.
    # See the discussion in this paper: https://arxiv.org/abs/1507.02105
    # These stars will likely be removed from the catalog.
    # https://astronomy.stackexchange.com/questions/26250/what-is-the-proper-interpretation-of-a-negative-parallax
    # https://gea.esac.esa.int/archive/documentation/GDR2/
    plx = result['Plx'] * 1.0 * u.mas
    distance = Distance(parallax=plx, allow_negative=True)

    c = SkyCoord(
        obstime = ref_time,
        ra = ra * u.mas, 
        dec = dec * u.mas, 
        distance=distance,
        pm_ra_cosdec = result['pmRA'], # mas/yr
        pm_dec = result['pmDE'], # mas/yr
    )

    coords_updated = c.apply_space_motion(new_obstime=update_time)

    updated_result = result.copy()
    updated_result["RAdeg"] = coords_updated.ra.deg
    updated_result["DEdeg"] = coords_updated.dec.deg


    #v.coordinates = c
    df = updated_result.to_pandas()
    df.sort_values('Hpmag', inplace=True)
    df.to_csv(output_file, index=False)