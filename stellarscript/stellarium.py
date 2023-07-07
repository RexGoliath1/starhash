import requests
import subprocess
from time import sleep
import yaml
import math
import numpy as np

# Ongoing TODOs
# Display Relevant magnitudes only
# Merge set logic into url / string?
# StelSkyDrawer.twinkleAmount

np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.8f}'.format})

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
        if fov is None:
            self.fov = fov

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
           print(f"Input jnow: {jnow_str} ; View jnow: {response.json().get('jNow')}")

    def get_actions(self):
        # Get a list of available action IDs:
        response = requests.get(self.url_main + self.url_actions)
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
            print(f"Julian date: {jd}")
        else:
            raise Exception("Status: ", response.status_code)

    def set_view_props(self):
        """ Remove various non-stars and phenomenon not viewable by camera """
        # Turn off all markers / 
        for prop in self.disabled_properties:
            print(f"Turning off {prop}")
            self.set_prop_id(prop)

    def set_magnitude_props(self):
        """" Set minimum magnitude based on camera """
        # Approximate, reference GIANT for validation sim
        customStarMagLimit = 7.5 + 5 * np.log(100 * self.aperture)
        gdict = {'StelSkyDrawer.customStarMagLimit': f"{customStarMagLimit:.1f}"}
        response = requests.get(self.url_main + self.url_prop, data= gdict)
        if response.ok: 
            response = requests.post(self.url_main + self.url_do_action, data = gdict)
            if not response.ok:
                print(response.status_code)
        else:
            print(response.status_code)

        customPlanetMagLimit = 7.5 + 5 * np.log(100 * self.aperture)
        gdict = {'StelSkyDrawer.customPlanetMagLimit': f"{customPlanetMagLimit}"}
        response = requests.get(self.url_main + self.url_prop, data= gdict)
        if response.ok: 
            response = requests.post(self.url_main + self.url_do_action, data = gdict)
            if not response.ok:
                print(response.status_code)
        else:
            print(response.status_code)

        customNebulaMagLimit = 7.5 + 5 * np.log(100 * self.aperture)
        gdict = {'StelSkyDrawer.customNebulaMagLimit': f"{customNebulaMagLimit}"}
        response = requests.get(self.url_main + self.url_prop, data= gdict)
        if response.ok: 
            response = requests.post(self.url_main + self.url_do_action, data = gdict)
            if not response.ok:
                print(response.status_code)
        else:
            print(response.status_code)

        gdict = {'StelSkyDrawer.flagDrawBigStarHalo': "false"}
        response = requests.get(self.url_main + self.url_prop, data= gdict)
        if response.ok: 
            response = requests.post(self.url_main + self.url_do_action, data = gdict)
            if not response.ok:
                print(response.status_code)
        else:
            print(response.status_code)


    def set_prop_id(self, prop):
        """ Set Stellarium property value """
        gdict = {'id': prop}
        
        response = requests.get(self.url_main + self.url_prop, data = gdict)

        if response.ok: 
            prop_value = response.json().get(prop).get('value')
            if(prop_value):
                response = requests.post(self.url_main + self.url_do_action, data = gdict)
                if not response.ok:
                    print(response.status_code)
        else:
            print(response.status_code)

    def set_prop_id(self, prop):
        """ Disable Stellarium property """
        gdict = {'id': prop}
        response = requests.get(self.url_main + self.url_prop, data = gdict)

        if response.ok: 
            prop_value = response.json().get(prop).get('value')
            if(prop_value):
                response = requests.post(self.url_main + self.url_do_action, data = gdict)
                if not response.ok:
                    print(response.status_code)
        else:
            print(response.status_code)


    def stop_process(self):
        self.proc_stellarium.kill()

    def run_scene(self):
        """ Set view to ra/dec """

        if not hasattr(self, "dec"):
            self.dec = 0
        
        for step in range(-step_size,step_size):
            sleep(delay_sec)
            dec = self.dec - np.pi / 2.0 * (step / step_size)
            ra = 0

            jnow = [math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)]
            jnow = np.array(jnow)
            jnow_str = np.array2string(jnow, separator=',')
            self.get_date()
            stel.set_boresight(jnow_str=jnow_str)

if __name__ == "__main__":
    config_file = "stellar_config.yaml"
    stel = Stellarium(config_file)
    step_size = stel.step_size
    delay_sec = stel.delay_sec
    print("Printing Date.... ")
    stel.set_date()
    stel.get_date()
    stel.set_fov()
    stel.set_view_props()
    stel.set_magnitude_props()
    stel.run_scene()
    stel.stop_process()
