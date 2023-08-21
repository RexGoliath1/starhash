import os
import requests
import subprocess
from time import sleep
import yaml
import math
import numpy as np
import json

# Ongoing TODOs
# StelSkyDrawer.twinkleAmount
# Output Frame data somewhere
# Disable fullscreen and set resolution appropriately

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
REPO_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "results")
STAR_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "stars")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(STAR_OUTPUT_DIRECTORY, exist_ok=True)
DEFAULT_JSON_OUTPUT= os.path.join(OUTPUT_DIRECTORY, "temp.json")
OUTPUT_HIPPARCOS = False

np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.8f}'.format})

# Put the Stellarium Application exec here
stellarium_path = r"/Users/stevengonciar/Downloads/Stellarium.app/Contents/MacOS/stellarium"

class Stellarium():
    def __init__(self, config):
        # Read in configuration and initialize Stellarium
        with open(config, 'r') as file:
            settings = yaml.safe_load(file)
            for k, v in settings.items():
                setattr(self, k, v)

        self.proc_stellarium = subprocess.Popen([stellarium_path], stdout=subprocess.PIPE, shell=True)

        for _ in range(20):
            try:
                response = requests.get(self.url_main + self.url_status) 
                break
            except:
                sleep(1)

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
        #param_view = {'altAz': jnow_str}
        response = requests.post(self.url_main + self.url_view, data = param_view)
        if not response.ok:
            print(response.status_code)
        response = requests.get(self.url_main + self.url_view) 
        if not response.ok:
            print(response.status_code)
        else:
            pass # TODO: Turn back on in some form
            print(f"Input jnow: {jnow_str} ; View jnow: {response.json().get('jNow')}, View j2000: {response.json().get('j2000')}")

    def get_actions(self):
        # Get a list of available action IDs:
        response = requests.get(self.url_main + self.url_actions)
        if response.ok:
            file_path = os.path.join(OUTPUT_DIRECTORY, "actions.json")
            with open(file_path, 'w') as fp:    
                json.dump(response.json(), fp, indent=4)
        return response.status_code


    def get_property(self):
        # Get a list of available action IDs:
        response = requests.get(self.url_main + self.url_prop)
        file_path = os.path.join(OUTPUT_DIRECTORY, "properties.json")
        with open(file_path, 'w') as fp:    
            json.dump(response.json(), fp, indent=4)
        return response.status_code


    def get_view(self):
        # Get a list of available action IDs:
        response = requests.get(self.url_main + self.url_view)
        file_path = os.path.join(OUTPUT_DIRECTORY, "view.json")
        with open(file_path, 'w') as fp:    
            json.dump(response.json(), fp, indent=4)
        return response.status_code


    def get_view_skyculture(self):
        # Get a list of available action IDs:
        response = requests.get(self.url_main + self.url_view_skyculture)
        file_path = os.path.join(OUTPUT_DIRECTORY, "url_view_skyculture.html")
        with open(file_path, 'w') as fp:    
            fp.write(response.text)
        return response.status_code


    def get_projection(self):
        # Get a list of available action IDs:
        response = requests.get(self.url_main + self.url_projection)
        file_path = os.path.join(OUTPUT_DIRECTORY, "projections.json")
        with open(file_path, 'w') as fp:    
            json.dump(response.json(), fp, indent=4)
        return response.status_code


    def get_simbad(self):
        # Get a list of available action IDs:
        response = requests.get(self.url_main + self.url_simbad)
        file_path = os.path.join(OUTPUT_DIRECTORY, "simbad.json")
        if response.ok:
            with open(file_path, 'w') as fp:    
                json.dump(response.json(), fp, indent=4)
        return response.status_code


    def get_objecttypes(self):
        # Get a list of available action IDs:
        response = requests.get(self.url_main + self.url_objecttypes)
        file_path = os.path.join(OUTPUT_DIRECTORY, "objecttypes.json")
        if response.ok:
            with open(file_path, 'w') as fp:    
                json.dump(response.json(), fp, indent=4)
        return response.status_code

    def get_listobjectsbytype(self, type="StarMgr"):
        # Get a list of available action IDs:
        gdict = {'type': type}
        response = requests.get(self.url_main + self.url_listobjectsbytype, data = gdict)

        file_path = os.path.join(OUTPUT_DIRECTORY, f"{type}.json")
        if response.ok:
            js = response.json()
            with open(file_path, 'w') as fp:    
                json.dump(js, fp, indent=4)
        return response.status_code


    def get_all_hip_stars(self):
        for ii in range(1, 118218 + 1):
            stel.get_objectinfo(obj_name=f"HIP {ii}")


    def get_objectinfo(self, obj_name="HIP 56572", obj_format="json"):
        # Get a list of available action IDs:
        gdict = {'name': obj_name, 'format': obj_format}
        response = requests.get(self.url_main + self.url_objectinfo, data = gdict)

        file_path = os.path.join(STAR_OUTPUT_DIRECTORY, f"{obj_name}.json")
        if response.ok:
            with open(file_path, 'w') as fp:    
                json.dump(response.json(), fp, indent=4)
        return response.status_code


    def set_date(self):
        """ Set time and timestep"""
        param_date = {'time': str(self.jday)}
        response = requests.post(self.url_main + self.url_time, data = param_date)
        if not response.ok:
            raise Exception("jday: ", response.status_code)

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
        magnitude_limit = 7.5 + 5 * np.log(100 * self.aperture)
        self.set_property("StelSkyDrawer.customStarMagLimit", f"{magnitude_limit:.1f}")
        self.set_property("StelSkyDrawer.customPlanetMagLimit", f"{magnitude_limit:.1f}")
        self.set_property("StelSkyDrawer.customNebulaMagLimit", f"{magnitude_limit:.1f}")


    def set_property(self, prop, value):
        """ Set Stellarium property value """
        gdict = {'id': prop, 'value': value}
        
        response = requests.get(self.url_main + self.url_prop, data = gdict)

        if response.ok: 
            # TODO: Figure out what to do for comparing current value...
            prop_value = response.json().get(prop).get('value')
            #if(prop_value):

            response = requests.post(self.url_main + self.url_prop_set, data = gdict)
            # response = requests.post(self.url_main + self.url_do_action, data = gdict)
            if not response.ok:
                print(response.status_code)
            else:
                response = requests.get(self.url_main + self.url_prop, data = gdict)
                print(f"This Response: {response.text}")
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


    def run_scene(self):
        """ Set view to ra/dec """

        if not hasattr(self, "dec"):
            self.dec = 0
        
        for step in range(-self.step_size, self.step_size):
            sleep(self.delay_sec)
            dec = self.dec - np.pi / 2.0 * (step / self.step_size)
            ra = 0

            jnow = [math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)]
            jnow = np.array(jnow)
            jnow_str = np.array2string(jnow, separator=',')
            self.get_date()
            stel.set_boresight(jnow_str=jnow_str)
            self.get_screenshot()

if __name__ == "__main__":
    config_file = "stellar_config.yaml"
    stel = Stellarium(config_file)
    stel.get_actions()
    stel.get_property()
    stel.get_view()
    stel.get_view_skyculture()
    stel.get_simbad()

    # TODO Group together better
    stel.get_objecttypes()
    stel.get_listobjectsbytype()
    if OUTPUT_HIPPARCOS:
        stel.get_all_hip_stars()

    stel.set_date()
    stel.get_date()
    stel.set_view_props()
    stel.set_fov()
    stel.run_scene()
    stel.stop_process()
