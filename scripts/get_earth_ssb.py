from astropy.coordinates import get_body_barycentric, EarthLocation
from astropy.time import Time
from astropy.constants import au
import numpy as np

gregorian_date = "2024-02-03 12:00:00.000"
# gregorian_date = "1991-04-02 03:56:23.317"
if gregorian_date is None:
    date = Time.now()
else:
    #date = Time(gregorian_date, scale="TT", format="iso")
    date = Time(gregorian_date)

# ICRS xyz earth position centered at SSB at current time
earth_pos_au = get_body_barycentric('earth', date)
earth_pos_km = earth_pos_au.xyz.to_value('km')
earth_pos_m = earth_pos_au.xyz.to_value('m')

# Convert AU to kilometers
print(f"Earth Position (AU): {earth_pos_au}")
# print(f"earth_pos_km : {earth_pos_km}")
# print(f"earth_pos_m : {earth_pos_m }")


print(f"Besselian Year: {date.byear}")


epoch_start = Time(1991.25, format='jyear', scale='tt')
time_difference = date.jyear - epoch_start.jyear
print(f"Time difference (for PM): {time_difference}")

epoch_start = Time(1991.25, format='byear', scale='tt')
print(f"epoch_start iso: {epoch_start.iso}")
