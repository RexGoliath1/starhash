from sys import platform
import os
import requests
import subprocess
import time
import yaml
import numpy as np
import json
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import itertools
import pandas as pd
import sqlite3
from astropy.time import Time

# Ongoing TODOs
# StelSkyDrawer.twinkleAmount

# Various output directories
CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
REPO_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "results", time.strftime("%Y%m%d-%H%M%S"))
STAR_OUTPUT_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "results", ".stars")
IMAGE_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "images")
POSITION_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "img_coords")
RA_DEC_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "ra_dec")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(STAR_OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(POSITION_OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(RA_DEC_OUTPUT_DIRECTORY, exist_ok=True)

# Enable both to output Hipparcos positions. Only needed once if using the same time
NUM_HIP_STARS = 118322 # tsv contains up to ID 120404?
# TODO: Remove stars that don't exist
HIP_NAMES = [f"HIP {ii}" for ii in range(1, NUM_HIP_STARS + 1)]
RECACHE_HIPPARCOS = False
# For each frame / J2000 Boresight location, output star locations
OUTPUT_POSITIONS = True
OUTPUT_RA_DEC = True
FILTER_VARIABLE_STARS = False

COLS_TO_CONVERT = {
    'above-horizon': 'bool',
    'absolute-mag': 'float',
    'airmass': 'float',
    'altitude': 'float',
    'altitude-geometric': 'float',
    'ambientInt': 'float',
    'ambientLum': 'float',
    'appSidTm': 'str',  # Time as string
    'azimuth': 'float',
    'azimuth-geometric': 'float',
    'bV': 'float',
    'dec': 'float',
    'decJ2000': 'float',
    'distance-ly': 'float',
    'elat': 'float',
    'elatJ2000': 'float',
    'elong': 'float',
    'elongJ2000': 'float',
    'found': 'bool',
    'glat': 'float',
    'glong': 'float',
    'hourAngle-dd': 'float',
    'hourAngle-hms': 'str',  # Time as string
    'iauConstellation': 'str',
    'localized-name': 'str',
    'meanSidTm': 'str',  # Time as string
    'name': 'str',
    'object-type': 'str',
    'parallax': 'float',
    'ra': 'float',
    'raJ2000': 'float',
    'rise': 'str',  # Time as string
    'rise-dhr': 'float',
    'set': 'str',  # Time as string
    'set-dhr': 'float',
    'sglat': 'float',
    'sglong': 'float',
    'size': 'int',
    'size-dd': 'float',
    'size-deg': 'str',  # Angle as string
    'size-dms': 'str',  # Angle as string
    'spectral-class': 'str',
    'star-type': 'str',
    'transit': 'str',  # Time as string
    'transit-dhr': 'float',
    'type': 'str',
    'variable-star': 'str',
    'vmag': 'float',
    'vmage': 'float',
    'wds-position-angle': 'int',
    'wds-separation': 'float',
    'wds-year': 'int',
    'period': 'float'  # Assuming period is numeric; change if it's a different format
}

# Debug variables
TQDM_SILENCE = True
DEG2RAD = np.pi / 180.0

np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.8f}'.format})

# Put the Stellarium Application exec here
if platform == "linux":
    stellarium_path = r"/home/parallels/git/stellarium/build/src/stellarium"
elif platform == "darwin":
    stellarium_path = r"/Users/stevengonciar/Downloads/Stellarium.app/Contents/MacOS/stellarium"
else:
    raise Exception("Only for my linux and mac")

stellarium_args = f' --screenshot-dir "{IMAGE_OUTPUT_DIRECTORY}"'

class Stellarium():
    def __init__(self, config):
        self.debug_stel = 0
        self.jnow = 0
        self.dec_start = 0
        self.dec_end = 0
        self.ra_start = 0
        self.ra_end = 0
        self.num_steps = 0
        self.url_main = ""
        self.url_status = ""
        self.url_do_action = ""
        self.url_prop = ""
        self.url_prop_set = ""
        self.url_fov = ""
        self.url_time = ""
        self.url_listobjectsbytype = ""
        self.url_objecttypes = ""
        self.url_objectinfo = ""
        self.url_view_skyculture = ""
        self.url_location = ""
        self.url_actions = ""
        self.url_view = ""
        self.url_projection = ""
        self.url_simbad = ""
        self.delay_sec = 0
        self.aperture = 0
        self.star_info = {}
        self.fx = 0
        self.fy = 0
        self.width = 0
        self.height = 0
        self.altitude = 0
        self.latitude = 0
        self.longitude = 0
        self.disabled_properties = []
        self.enabled_properties = []
        self.timerate = 0
        self.gregorian_date = 0
        self.julian_date = 0
        self.R = 0
        self.E = 0
        self.K = 0
        # For debug
        self.r_j2000_to_bore = None
        self.r_bore_to_camera = None
        self.r = None

        # Read in configuration and initialize Stellarium class
        with open(config, 'r') as file:
            settings = yaml.safe_load(file)
            for k, v in settings.items():
                setattr(self, k, v)

        self.proc_stellarium = subprocess.Popen([stellarium_path + stellarium_args], stdout=subprocess.PIPE, shell=True)

        # Once Stellarium starts, attempt to query server
        response = None
        for _ in range(20):
            try:
                response = requests.get(self.url_main + self.url_status) 
                break
            except:
                time.sleep(1)

        if response is None:
            raise Exception(f"No response from Stellarium server {self.url_main + self.url_status}")

        if not response.ok:
            raise Exception("Status: ", response.status_code)

    def set_fov(self, fov = None):
        if fov is not None:
            self.fov = fov

        assert(not self.fov is None)

        param_fov = {'fov': str(self.fov)}

        response = requests.post(self.url_main + self.url_fov, data = param_fov)
        return response.status_code

    def set_boresight(self, jnow_str):
        assert isinstance(jnow_str, str)
        param_view = {'jNow': jnow_str}
        response = requests.post(self.url_main + self.url_view, data = param_view)
        if not response.ok:
            print(response.status_code)
        response = requests.get(self.url_main + self.url_view) 
        if not response.ok:
            print(response.status_code)
        else:
            pass # TODO: Turn back on in some form
            print(f"View jnow: (ra, dec) = ({self.ra}, {self.dec});")
            print(f"    [x,y,z] = {response.json().get('jNow')}")
            print(f"    jnow_str = {jnow_str}")

    def get_actions(self):
        # Get a list of available action IDs:
        response = requests.get(self.url_main + self.url_actions)
        if response.ok:
            file_path = os.path.join(OUTPUT_DIRECTORY, "actions.json")
            with open(file_path, 'w') as fp:    
                json.dump(response.json(), fp, indent=4)
        return response.status_code


    def get_property(self):
        """ TODO Debugging: Get a list of available properties """
        response = requests.get(self.url_main + self.url_prop)
        file_path = os.path.join(OUTPUT_DIRECTORY, "properties.json")
        with open(file_path, 'w') as fp:    
            json.dump(response.json(), fp, indent=4)
        return response.status_code


    def get_view(self):
        """ TODO Debugging: Get a list of available actions """
        response = requests.get(self.url_main + self.url_view)
        file_path = os.path.join(OUTPUT_DIRECTORY, "view.json")
        with open(file_path, 'w') as fp:    
            json.dump(response.json(), fp, indent=4)
        return response.status_code


    def get_view_skyculture(self):
        """ TODO Debugging: Get a list of available sky cultures """
        response = requests.get(self.url_main + self.url_view_skyculture)
        file_path = os.path.join(OUTPUT_DIRECTORY, "url_view_skyculture.html")
        with open(file_path, 'w') as fp:    
            fp.write(response.text)
        return response.status_code


    def get_projection(self):
        """ Debugging: Get a list of available projections """
        response = requests.get(self.url_main + self.url_projection)
        file_path = os.path.join(OUTPUT_DIRECTORY, "projections.json")
        with open(file_path, 'w') as fp:    
            json.dump(response.json(), fp, indent=4)
        return response.status_code


    def get_simbad(self):
        """ TODO: Debugging: Get a list of available simbad Types """
        response = requests.get(self.url_main + self.url_simbad)
        file_path = os.path.join(OUTPUT_DIRECTORY, "simbad.json")
        if response.ok:
            with open(file_path, 'w') as fp:    
                json.dump(response.json(), fp, indent=4)
        return response.status_code


    def get_obj_types(self):
        """ Debugging: Get a list of available object types """
        response = requests.get(self.url_main + self.url_objecttypes)
        file_path = os.path.join(OUTPUT_DIRECTORY, "objecttypes.json")
        if response.ok:
            with open(file_path, 'w') as fp:    
                json.dump(response.json(), fp, indent=4)
        return response.status_code


    def get_obj_list(self, type="StarMgr"):
        """ Debugging: Get a list of available objects by type """
        # Get a list of available action IDs:
        gdict = {'type': type}
        response = requests.get(self.url_main + self.url_listobjectsbytype, data = gdict)

        file_path = os.path.join(OUTPUT_DIRECTORY, f"{type}.json")
        if response.ok:
            js = response.json()
            with open(file_path, 'w') as fp:    
                json.dump(js, fp, indent=4)
        return response.status_code


    def get_star_positions(self):
        """ Get star position in camera using the Intrinsic + Extrinsic Transformation matrix """

        positions = {}
        coords = {}
        magnitude_limit = 7.0 + 5 * np.log10(100 * self.aperture)

        # For all stars, get jNow and convert to pixel postion
        for obj_name in tqdm(iterable=HIP_NAMES, desc="Getting Star Positions", disable=TQDM_SILENCE):

            # First, check if file is available, otherwise query
            file_path = os.path.join(STAR_OUTPUT_DIRECTORY, f"{obj_name}_{self.gregorian_date}.json")
            if os.path.exists(file_path):
                with open(file_path) as fp:
                    data = json.load(fp)
            else:
                data = self.get_obj_info(obj_name=obj_name)

            if data is None:
                continue

            # If the star is lower in magnitude than requested, ignore it
            if data["vmag"] > magnitude_limit:
                continue

            if FILTER_VARIABLE_STARS and data["variable-star"] != "no":
                continue # pulsating, variable, rotating, eclipsing-binary, eruptive, cataclysmic

            (ra, dec) = (data["ra"] * DEG2RAD, data["dec"] * DEG2RAD)
            [x, y, z] = self.j2000_to_xyz(ra, dec)
            x_w = np.array([[x, y, z, 1]])
            x_c = np.matmul(self.E, x_w.T)
            x_p = np.matmul(self.K, x_c)
            uv = (x_p / x_p[2][0])[:-1]
            uv = list(itertools.chain(*uv))

            if x_p[2][0] < 0:
                continue

            # For extra debug only
            #if abs(self.ra - ra) < 0.01 and abs(self.dec - dec) < 0.01:
            #    print("Should be pretty close  ....")

            # For debugging, print out stars within frame
            if 0 <= uv[0] <= self.width and 0 <= uv[1] <= self.height:
                print(f"{obj_name}: IN FRAME @ ({uv})")
                coords[obj_name] = uv
                positions[obj_name] = [x, y, z]

        return positions, coords


    def load_star_info(self):
        """ Load star info from cache or create cache """
        file_path = os.path.join(STAR_OUTPUT_DIRECTORY, f"{self.gregorian_date}.db")

        # First, if we are recaching, delete the file
        if RECACHE_HIPPARCOS:
            if os.path.exists(file_path):
                os.remove(file_path)

        if os.path.exists(file_path):
            # Connect to SQLite database
            conn = sqlite3.connect(file_path)
            # SQL query to select all data from the 'stars' table
            query = "SELECT * FROM stars"
            # Execute the query and load the result into a DataFrame
            df = pd.read_sql_query(query, conn)
            df.set_index('index', inplace=True)
            self.star_info = df.to_dict(orient='index')
        else:
            for obj_name in tqdm(iterable=HIP_NAMES, desc="Caching Hipparcos Star Info", disable=TQDM_SILENCE):
                self.get_obj_info(obj_name=obj_name, output=True)
            # Rewrite the entire file again (JSON is terrible)
            df = pd.DataFrame(self.star_info)
            df = df.transpose()
            for column, dtype in COLS_TO_CONVERT.items():
                if dtype == 'float':
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                elif dtype == 'int':
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')  # 'Int64' (capital I) to handle NaN
                elif dtype == 'bool':
                    df[column] = df[column].astype(bool)
                elif dtype == 'str':
                    df[column] = df[column].astype(str)
            # Connect to SQLite database
            conn = sqlite3.connect(file_path)
            # Write the DataFrame to SQLite table 'stars'
            df.to_sql('stars', conn, if_exists='replace', index=True)

        # Close the connection
        conn.close()

    def get_obj_info(self, obj_name="HIP 56572", output=False):
        """ Return and cache stellarium object info """

        # Check if star_info contains the object
        if obj_name in self.star_info.keys():
            objinfo = self.star_info[obj_name]

            # Do this to avoid querying server again
            if objinfo == "Not Found":
                objinfo = None
            return objinfo
        
        # Query the server for the object info
        gdict = {'name': obj_name, 'format': "json"}
        response = requests.get(self.url_main + self.url_objectinfo, data = gdict)

        if response.ok:
            # Append to the star info dictionary
            star_dict = {obj_name: response.json()}
            self.star_info.update(star_dict)
            return response.json()
        else:
            if response.text == "object name not found":
                star_dict = {obj_name: "Not Found"}                
                self.star_info.update(star_dict)
            print(f"{obj_name}: {response.text}")
            return None

    def set_date(self):
        """ Set time and timestep"""

        # Convert from Gregorian Input to Julian Date Stellarium input
        time_obj = Time(self.gregorian_date)
        self.julian_date = time_obj.jd

        param_date = {'time': str(self.julian_date)}
        response = requests.post(self.url_main + self.url_time, data = param_date)
        if not response.ok:
            raise Exception("julian_date: ", response.status_code)

        param_date = {'timerate': str(self.timerate)}
        response = requests.post(self.url_main + self.url_time, data = param_date)
        if not response.ok:
            raise Exception("timerate: ", response.status_code)

    def set_location(self):
        """ Set LLA """
        param_lat = {'latitude': str(self.latitude)}
        response = requests.post(self.url_main + self.url_location, data = param_lat)
        if not response.ok:
            raise Exception("lat: ", response.status_code)
        
        param_long = {'longitude': str(self.longitude)}
        response = requests.post(self.url_main + self.url_location, data = param_long)
        if not response.ok:
            raise Exception("longitude: ", response.status_code)

        param_alt = {'altitude': str(self.altitude)}
        response = requests.post(self.url_main + self.url_location, data = param_alt)
        if not response.ok:
            raise Exception("lat: ", response.status_code)


    def get_date(self):
        """ Get Simulation Date """
        response = requests.get(self.url_main + self.url_status) 

        if response.ok:
            jd = response.json().get('time').get('jday')
            #print(f"Julian date: {jd}")
        else:
            raise Exception("Status: ", response.status_code)

    def set_view_props(self):
        """ Remove various non-stars and phenomenon not viewable by camera """
        # Turn off all markers /
        for prop in self.disabled_properties:
            print(f"Turning off {prop}")
            self.set_property(prop, "false")

        for prop in self.enabled_properties:
            print(f"Turning on {prop}")
            self.set_property(prop, "true")

        # Set Projection
        self.set_property('StelCore.currentProjectionType', 'ProjectionPerspective')

        # Set minimum magnitude based on camera. Approximate, reference GIANT for validation sim
        magnitude_limit = 7.0 + 5 * np.log10(100 * self.aperture)
        self.set_property("StelSkyDrawer.customStarMagLimit", f"{magnitude_limit:.1f}")
        self.set_property("StelSkyDrawer.customPlanetMagLimit", f"{magnitude_limit:.1f}")
        self.set_property("StelSkyDrawer.customNebulaMagLimit", f"{magnitude_limit:.1f}")

        # Set Absolute + Relative Star Scale for display
        # self.set_property("StelSkyDrawer.absoluteStarScale", f"{1.25}")
        # self.set_property("StelSkyDrawer.relativeStarScale", f"{0.5}")

        # Set screen shot custom width (this might not be right, may need to actually adjust screen. fuck)
        self.set_property("MainView.customScreenshotHeight", f"{int(self.height)}")
        self.set_property("MainView.customScreenshotWidth", f"{int(self.width)}")


    def set_property(self, prop, value):
        """ Set Stellarium property value """
        gdict = {'id': prop, 'value': value}
        response = requests.get(self.url_main + self.url_prop, data = gdict)

        if response.ok: 
            response = requests.post(self.url_main + self.url_prop_set, data = gdict)
            if not response.ok or ("error: unknown property" in response.text):
                print(f"Property Set {prop} Failed: {response.status_code} Text: {response.text}")
        else:
            print(response.status_code)

    def get_screenshot(self):
        """" Set scene projection """
        gdict = {'id': "actionSave_Screenshot_Global"}
        response = requests.post(self.url_main + self.url_do_action, data = gdict)
        if not response.ok:
            print(response.status_code)


    def stop_process(self):
        """ Stop the stellarium app """
        self.proc_stellarium.kill()

    def j2000_to_xyz(self, ra, dec):
        """ Convert RA/DEC to Celestial Sphere coordinates """
        # https://stellarium.org/doc/head/remoteControlApi.html#rcMainServiceViewGet
        return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])

    def get_intrinsic(self):
        """ Get Intrinsic Matrix """
        # Not sure if this is correct, would have to figure out Ocular plugin
        # Does match output of Astrometry
        
        ar = 4/3 # TODO: Can't seem to set this along with image size
        fx = (self.width / 2) / np.tan(DEG2RAD * ar * self.fov / 2)
        fy = (self.height / 2) / np.tan(DEG2RAD * self.fov / 2)
        cx = self.width / 2.0
        cy = self.height / 2.0

        self.K = np.array([
            [fx,    0.0,    cx],
            [0.0,   fy,     cy],
            [0.0,   0.0,    1.0]
        ])

    def get_extrinsic(self):
        """ Get Intrinsic + Extrinsic Transformation matrix """
        # Assuming equatorial mount reduces roll to zero ...
        self.r_j2000_to_bore = Rotation.from_euler('ZYX', [-self.ra, self.dec, 0.0], degrees=False)
        #self.r_bore_to_camera = Rotation.from_euler('zyx', [-90.0, 0.0, -90.0], degrees=True)
        self.r_bore_to_camera = Rotation.from_euler('zyx', [90.0, 0.0, 90.0], degrees=True)
        #self.r = self.r_j2000_to_bore * self.r_bore_to_camera
        self.r =  self.r_bore_to_camera * self.r_j2000_to_bore
        #self.R = np.matmul(r_j2000_to_bore.as_matrix(), r_bore_to_camera.as_matrix())
        self.R = self.r.as_matrix()
        t = np.array([[0, 0, 0]])
        #self.R = np.linalg.inv(self.R)
        self.E = np.concatenate((self.R, t.T), axis=1)


    def run_debug(self):
        """ Run debugging to output jsons of various parameters """
        self.get_actions()
        self.get_property()
        self.get_view()
        self.get_view_skyculture()
        self.get_simbad()
        self.get_obj_types()
        self.get_obj_list()
        self.get_date()

    def run_scene(self):
        """ Set view to ra/dec """

        # Here are a few debugging functions for outputting Stellarium configurable params
        if self.debug_stel:
            self.run_debug()
        
        # First, configure a few of view parameters based on input YAML
        self.set_date()
        self.load_star_info()
        self.set_view_props()
        self.set_fov()

        # Get the camera intrisincs for later projection
        self.get_intrinsic()

        # Now, loop through requested RA/DEC. This is the center of the boresight on EQ mount.
        # TODO: Check: EQ Mount aligned with earth rotation, thus no roll and dec/ra corresponds to pitch/yaw
        # Meaning, if we know RA/DEC we can get RPY -> Extrinsic Matrix -> Star pixel positions
        dec = np.linspace(self.dec_start, self.dec_end, self.num_steps)
        ra = np.linspace(self.ra_start, self.ra_end, self.num_steps)

        for ii in range(0, self.num_steps):
            self.dec = dec[ii]
            self.ra = ra[ii]
            self.jnow = self.j2000_to_xyz(self.ra, self.dec)
            jnow_str = np.array2string(self.jnow, separator=',')

            self.set_boresight(jnow_str=jnow_str)
            # Wait, neccessary for rendering
            time.sleep(self.delay_sec)

            self.get_extrinsic()

            if OUTPUT_POSITIONS:
                positions, coords = self.get_star_positions()
                coords_file = os.path.join(POSITION_OUTPUT_DIRECTORY, f"{ii}.json")
                with open(coords_file, 'w') as fp:
                    json.dump(coords, fp)

                pos_file = os.path.join(POSITION_OUTPUT_DIRECTORY, f"positions_{ii}.json")
                with open(pos_file, 'w') as fp:
                    json.dump(positions, fp)
            
            if OUTPUT_RA_DEC:
                # Output RA/DEC for validation
                ra_dec_file = os.path.join(RA_DEC_OUTPUT_DIRECTORY, f"{ii}_ra_dec.json")
                with open(ra_dec_file, 'w') as fp:    
                    json.dump({
                        "ra": self.ra, 
                        "dec": self.dec, 
                        "E": self.E.tolist(), 
                        "K": self.K.tolist()
                    }, fp)

            # This script outputs pictures in local user folder. 
            # afaik this folder is not configurable through API.
            self.get_screenshot()


if __name__ == "__main__":
    config_file = os.path.join(CURRENT_DIRECTORY, "stellar_config.yaml")
    stel = Stellarium(config_file)


    stel.run_scene()
    stel.stop_process()
