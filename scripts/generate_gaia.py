from astropy.coordinates import SkyCoord 
from astroquery.gaia import Gaia
import astropy.units as u
import matplotlib.pyplot as plt

coord = SkyCoord(ra=0.0, dec=0.0, unit=(u.degree, u.degree), frame='icrs')
width = u.Quantity(10.0, u.deg)
height = u.Quantity(10.0, u.deg)

r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
df = r.to_pandas()
x = df['ra'].dropna().values
y = df['dec'].dropna().values
plt.scatter(x, y, s=0.1)
plt.show()
r.pprint(max_lines=12, max_width=130)