from astropy.coordinates import get_body_barycentric, EarthLocation
from astropy.time import Time
from astropy.constants import au
import numpy as np

hip_epoch       = "2000-01-01 12:00:00.000" # HIP2 J2000 Data is at J2000
gregorian_date  = "2024-02-03 12:00:00.000"
# gregorian_date = "1991-04-02 03:56:23.317"
if gregorian_date is None:
    date = Time.now()
    hip_date = Time(hip_epoch)
else:
    #date = Time(gregorian_date, scale="TT", format="iso")
    date = Time(gregorian_date)
    hip_date = Time(hip_epoch)

# ICRS xyz earth position centered at SSB at launch
earth_pos_au = get_body_barycentric('earth', date)
earth_pos_km = earth_pos_au.xyz.to_value('km')
earth_pos_m = earth_pos_au.xyz.to_value('m')

# ICRS xyz earth position centered at SSB at catalog epoch
cat_earth_pos_au = get_body_barycentric('earth', hip_date)
cat_earth_pos_km = earth_pos_au.xyz.to_value('km')
cat_earth_pos_m = earth_pos_au.xyz.to_value('m')

# Convert AU to kilometers
print(f"Launch Earth Position (AU): {earth_pos_au}")
# print(f"Launch earth_pos_km : {earth_pos_km}")
# print(f"Launch earth_pos_m : {earth_pos_m }")

print(f"Catalog Earth Position (AU): {cat_earth_pos_au}")
# print(f"Catalog earth_pos_km : {cat_earth_pos_km}")
# print(f"Catalog earth_pos_m : {cat_earth_pos_m }")


print(f"Julian Year of Launch: {date.jyear}")
print(f"Besselian Year of Launch: {date.byear}")
print(f"Julian Year of Catalog: {hip_date.jyear}")
print(f"Besselian Year of Catalog: {hip_date.byear}")


epoch_start = Time(1991.25, format='jyear', scale='tt')
time_difference = date.jyear - epoch_start.jyear
print(f"Time difference (for PM): {time_difference}")
