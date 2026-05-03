import numpy as np
import mne
from mne.preprocessing.nirs import optical_density, beer_lambert_law


class extract_fnirs_trials_class:
    
    def extract_fnirs_trials(folder_path):
        print(f"Processing data from: {folder_path}")
        
        raw_intensity = mne.io.read_raw_nirx(folder_path, preload=True)
        raw_od = optical_density(raw_intensity)
        raw_haemo = beer_lambert_law(raw_od, ppf=0.1)
        raw_haemo.filter(0.01, 0.1, h_trans_bandwidth=0.01, l_trans_bandwidth=0.005)
        
        # 4. Extract the trigger events
        events, event_dict = mne.events_from_annotations(raw_haemo)
        
        # NEW STEP: Print the dictionary so you can see the exact names of the triggers!
        print("Found these triggers:", event_dict)
        
        # 5. Define EXACTLY which triggers you want. 
        # (Note: You will need to check the printed event_dict to see the exact 
        # string names MNE gives to your triggers, e.g., '1.0', '2.0', or 'Stimulus/A')
        
        # Example: assuming the triggers for looking at the picture are 1 and 2
        target_events = {'Picture_Recognition_1': 1, 'Picture_Recognition_2': 2} 
        
        # 6. Slice the data using ONLY the target_events
        tmin, tmax = -2.0, 2.0 
        epochs = mne.Epochs(raw_haemo, events, 
                            event_id=target_events, # <--- THIS IS THE CRITICAL ADDITION
                            tmin=tmin, tmax=tmax, 
                            baseline=(tmin, 0), preload=True)
        
        fnirs_tensor = epochs.get_data()
        return fnirs_tensor



