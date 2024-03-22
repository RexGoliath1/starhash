import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord 
from astropy.units import Quantity
from astroquery.gaia import Gaia
import numpy as np
import matplotlib.pyplot as plt
from time import time


tables = Gaia.load_tables(only_names = True)
for table in tables:
    name = table.get_qualified_name()
    if name.find("gaiadr3") >= 0:
        print(name)

query = "SELECT gaiadr3.gaia_source.ra, gaiadr3.gaia_source.dec \
    FROM gaiadr3.gaia_source \
    "

query = "SELECT * \
FROM gaiadr1.gaia_source \
WHERE CONTAINS(POINT(gaiadr1.gaia_source.ra,gaiadr1.gaia_source.dec),CIRCLE(56.75,24.1167,2))=1;" \

coord = SkyCoord(ra=0.0, dec=0.0, unit=(u.degree, u.degree), frame='icrs')
width = u.Quantity(10.0, u.deg)
height = u.Quantity(10.0, u.deg)

cols = ["gaiadr3.gaia_source.ra", "gaiadr3.gaia_source.dec "]
Gaia.ROW_LIMIT = 1000
meta = Gaia.load_table("gaiadr3.gaia_source")
print(meta)

if meta is None:
    print("Null query")
    exit(-1)

for column in meta.columns:
    print(column.name)

# t1 = time()
# r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
# t2 = time()
# print(f"Job took {t2 - t1:.2f} seconds")

#r = Gaia.query_object_async(query)
# if r is None:
#     print("nothing found")
# else:
#     df = r.to_pandas()
#     print(df.shape)
# x = df['ra'].dropna().values
# y = df['dec'].dropna().values
# plt.scatter(x, y, s=0.1)
# plt.show()
# r.pprint(max_lines=12, max_width=130)
