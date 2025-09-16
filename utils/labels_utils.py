import numpy as np


def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'Walking'
    if label == 'FIX_running':
        return 'Running'
    if label == 'LOC_main_workplace':
        return 'At main workplace'
    if label == 'OR_indoors':
        return 'Indoors'
    if label == 'OR_outside':
        return 'Outside'
    if label == 'LOC_home':
        return 'At home'
    if label == 'FIX_restaurant':
        return 'At a restaurant'
    if label == 'OR_exercise':
        return 'Exercise'
    if label == 'LOC_beach':
        return 'At the beach'
    if label == 'OR_standing':
        return 'Standing'
    if label == 'WATCHING_TV':
        return 'Watching TV'
    if label == 'OR_outside':
        return 'Outside'
    if label == 'OR_indoors':
        return 'Indoors'
    
    if label.endswith('_'):
        label = label[:-1] + ')'
        pass
    
    label = label.replace('__',' (').replace('_',' ')
    label = label[0] + label[1:].lower()
    label = label.replace('i m','I\'m')
    return label

# Interpret the feature names to figure out for each feature what is the sensor it was extracted from
def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names])
    for (fi,feat) in enumerate(feature_names):
        if feat.startswith('raw_acc'):
            feat_sensor_names[fi] = 'Acc'
            pass
        elif feat.startswith('proc_gyro'):
            feat_sensor_names[fi] = 'Gyro'
            pass
        elif feat.startswith('raw_magnet'):
            feat_sensor_names[fi] = 'Magnet'
            pass
        elif feat.startswith('watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc'
            pass
        elif feat.startswith('watch_heading'):
            feat_sensor_names[fi] = 'Compass'
            pass
        elif feat.startswith('location'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith('location_quick_features'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith('audio_naive'):
            feat_sensor_names[fi] = 'Aud'
            pass
        elif feat.startswith('audio_properties'):
            feat_sensor_names[fi] = 'AP'
            pass
        elif feat.startswith('discrete'):
            feat_sensor_names[fi] = 'PS'
            pass
        elif feat.startswith('lf_measurements'):
            feat_sensor_names[fi] = 'LF'
            pass
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat)
        pass

    return feat_sensor_names

def new_label_name(label):
    mapping = {
        'OR_standing': 'Standing',
        'LYING_DOWN': 'Lying down',
        'SITTING': 'Sitting',
        'FIX_walking': 'Walking',
        'FIX_running': 'Running',
        'BICYCLING': 'Bicycling',
        'SLEEPING': 'Sleeping',
        'LAB_WORK': 'Lab work',
        'IN_CLASS': 'In class',
        'IN_A_MEETING': 'In meeting',
        'LOC_main_workplace': 'At main workplace',
        'IN_A_CAR': 'In car',
        'ON_A_BUS': 'On bus',
        'DRIVE_-_I_M_THE_DRIVER': 'Driving:Driver',
        'DRIVE_-_I_M_A_PASSENGER': 'Driving:Passenger',
        'LOC_home': 'At home',
        'FIX_restaurant': 'At a restaurant',
        'OR_exercise': 'Exercise',
        'COOKING': 'Cooking',
        'SHOPPING': 'Shopping',
        'STROLLING': 'Strolling',
        'DRINKING__ALCOHOL_': 'Drinking alcohol',
        'BATHING_-_SHOWER': 'Bathing:Shower',
        'CLEANING': 'Cleaning',
        'DOING_LAUNDRY': 'Doing laundry',
        'WASHING_DISHES': 'Washing dishes',
        'WATCHING_TV': 'Watching TV',
        'AT_A_PARTY': 'At a party',
        'AT_A_BAR': 'At a bar',
        'LOC_beach': 'At the beach',
        'SINGING': 'Singing',
        'TALKING': 'Talking',
        'COMPUTER_WORK': 'Computer Work',
        'EATING': 'Eating',
        'TOILET': 'Toilet',
        'GROOMING': 'Grooming',
        'DRESSING': 'Dressing',
        'AT_THE_GYM': 'At the gym',
        'STAIRS_-_GOING_UP': 'Stairs:up',
        'STAIRS_-_GOING_DOWN': 'Stairs:down',
        'ELEVATOR': 'Elevator',
        'AT_SCHOOL': 'At school',
        'OR_indoors': 'Indoors',
        'OR_outside': 'Outside'
    }

    return mapping.get(label, "Excluded")