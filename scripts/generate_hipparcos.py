# This script generates a CSV file containing the Hipparcos catalog
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord, Distance
import astropy.units as u
from astropy.time import Time
import os
import numpy as np
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import animation
import re
import pandas as pd

plot_kde = False
plot_rade = False
plot_rade_change = False

assert int(plot_kde) + int(plot_rade) + int(plot_rade_change) <= 1

def set_sky_coord(result, ref_time):

    plx = u.Quantity(result['Plx'])
    distance = Distance(parallax=plx, allow_negative=False)
    coord = SkyCoord(
        frame = "icrs",
        obstime = ref_time,
        ra = result["RArad"],
        dec = result["DErad"],
        distance=distance,
        # distance=result['Plx'],
        pm_ra_cosdec = result['pmRA'], # mas/yr
        pm_dec = result['pmDE'], # mas/yr
    )
    return coord

# set the reference time for proper motion calculations
ref_time = Time('2000-01-01 12:00:00.000')

# set the date to calculate the updated positions (Use now for testing)
# update_time = Time.now()
# update_time = Time('2005-01-01')
# update_time = Time('2024-02-03 12:00:00.000')
# update_time = Time('2022-09-11T17:55:01', format='isot', scale='tcb')
update_time = Time('2024-02-25T17:59:34', format='isot', scale='tcb')
# update_time = Time('2000-01-01 12:00:00.000')

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
    "B-V",
    "e_Plx",
    "e_pmRA",
    "e_pmDE",
    "sHp",
    "e_B-V",
    "V-I"
]

ra = 0.0
dec = 0.0

v = Vizier(catalog=catalog, columns=columns, row_limit=-1)

# result = v.query_constraints(Hpmag='>0.0')
# TODO: Look into using these. For now we will drop the use both here and in the catalog generator
result = v.query_constraints(Plx='>0.0')

if result is None:
    raise ValueError('No results found')
else:
    assert len(result) == 1
    result = result[0]
    ref_time_string = re.findall("\d+\.\d+", result["RArad"].description)[0]
    ref_time = Time(ref_time_string, format='jyear', scale='tcb')
    # ref_time = Time('2000-01-01')

    # There are apparently some negative parallaxes in the Hipparcos and Gaia catalogs. 
    # These are converted to NaN distances.
    # See the discussion in this paper: https://arxiv.org/abs/1507.02105
    # These stars will likely be removed from the catalog.
    # https://astronomy.stackexchange.com/questions/26250/what-is-the-proper-interpretation-of-a-negative-parallax
    # https://github.com/agabrown/astrometry-inference-tutorials/blob/master/luminosity-calibration/DemoNegativeParallax.ipynb
    # https://gea.esac.esa.int/archive/documentation/GDR2/
    
    #plx = result['Plx'] * 1.0 * u.mas
    #distance = Distance(parallax=plx, allow_negative=True)

    c = set_sky_coord(result, ref_time)

    updated_result = result.copy()
    updated_result["RAdeg"] = c.icrs.ra.deg
    updated_result["DEdeg"] = c.icrs.dec.deg

    coords_updated = c.apply_space_motion(new_obstime=update_time)

    # Trying to match original tsv...
    # updated_result["RArad"] = coords_updated.fk5.ra.deg
    # updated_result["DErad"] = coords_updated.fk5.dec.deg
    updated_result["RAJ2000"] = coords_updated.fk5.ra.deg
    updated_result["DEJ2000"] = coords_updated.fk5.dec.deg
    updated_result["RAICRS"] = coords_updated.icrs.ra.deg
    updated_result["DEICRS"] = coords_updated.icrs.dec.deg
    # updated_result["_RAJ2000"] = c.fk5.ra.deg
    # updated_result["_DEJ2000"] = c.fk5.dec.deg


    #v.coordinates = c
    df = updated_result.to_pandas()

    # df.sort_values('Hpmag', inplace=True)
    df.sort_values('HIP', inplace=True)
    df.to_csv(output_file, index=False, header=True)

    if plot_kde:
        # Make a KDE plot of RA and Dec
        x = df['RArad'].dropna().values
        y = df['DErad'].dropna().values
        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        fig = plt.figure(figsize=(7,8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        # alpha=0.5 will make the plots semitransparent
        ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)
        ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)

        ax1.set_xlim(x.min(), x.max())
        ax1.set_ylim(y.min(), y.max())
        ax2.set_xlim(x.min(), x.max())
        ax2.set_ylim(y.min(), y.max())
        plt.show()
    elif plot_rade:
        x = df['RArad'].dropna().values
        y = df['DErad'].dropna().values

        ax1 = plt.subplot(2, 1, 1)
        plt.subplot(211)
        plt.scatter(x, y, s=0.1)
        plt.xlim(0.0, 360.0)
        plt.ylim(-90.0, 90.0)
        plt.title(f"Scatter plot of RA and Dec in Hipparcos at time {update_time}")
        plt.subplot(212)
        plt.hist(df["Hpmag"], bins=100,edgecolor = "black", cumulative=True, log=True)
        plt.title("Histogram of Hipparcos Magnitudes (log scale)")
        plt.show()
    elif plot_rade_change:
        # Loop through time every year from 2000 to 2020
        # Calculate the change in RA and Dec for each star and plot as an animation
        start_year = 1000
        #end_year = 2000
        end_year = 9900
        skip_years = 100
        df = result.to_pandas()

        label_stars = True
        limit_fov = True

        if limit_fov:
            x_center = 180.0
            x_fov = 10.0
            x_min = x_center - x_fov / 2.0
            x_max = x_center + x_fov / 2.0

            y_center = 0.0
            y_fov = 10.0
            y_min = y_center - y_fov / 2.0
            y_max = y_center + y_fov / 2.0

            assert(x_min >= 0.0)
            assert(x_max <= 360.0)
            assert(y_min >= -90.0)
            assert(y_max <= 90.0)
        
        min_brightness = 0.0
        max_brightness = 7.5

        scatter_size = 3.0

        x = df['RArad'].dropna().values
        y = df['DErad'].dropna().values

        # Plot the first year
        fig, ax = plt.subplots(figsize=(16, 9), dpi=1920/16)
        cm = plt.cm.get_cmap('Reds_r')

        def animate(year):
            print(f"Year = {year}")
            time = Time(f"{year}-01-01")
            c = set_sky_coord(result, ref_time)
            coords_updated = c.apply_space_motion(new_obstime=time)

            # Set up the updated result            
            updated_result = result.copy()
            updated_result["RAdeg"] = coords_updated.ra.deg
            updated_result["DEdeg"] = coords_updated.dec.deg
            df = updated_result.to_pandas()
            df = df[df['RAdeg'].notna()]
            df = df[df['DEdeg'].notna()]
            df = df[df['Hpmag'] >= min_brightness]
            df = df[df['Hpmag'] <= max_brightness]

            # Get only stars in current FOV
            x = df['RAdeg'].values
            y = df['DEdeg'].values
            if limit_fov:
                in_fov = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
                df = df[in_fov]
                x = df['RAdeg'].values
                y = df['DEdeg'].values

            labels = "HIP_" + df["HIP"].dropna().astype(str)
            brightness = df["Hpmag"].dropna().values


            fig.clear()
            ax = fig.add_subplot(111)

            if limit_fov:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_xlim(0.0, 360.0)
                ax.set_ylim(-90.0, 90.0)

            ax.set_xlabel('Right Ascension [deg]')
            ax.set_ylabel('Declination [deg]')
            sc = ax.scatter(x, y, c=brightness,s=scatter_size, cmap=cm, vmin=min_brightness, vmax=max_brightness)
            plt.colorbar(sc)
            if label_stars:
                for ii, label in enumerate(labels):
                    ax.annotate(label, (x[ii], y[ii]))

            ax.set_title(f"Time: [{time}] HIP RA/DEC. Number of stars: {df.shape[0]}")

        ani = animation.FuncAnimation(fig, animate, interval=100, frames=list(range(start_year, end_year, skip_years)))
        ani.save('animation.gif', writer='pillow')

empty_entries = np.where(pd.isnull(df))
print(f"Done! Empty entries: {empty_entries}")
