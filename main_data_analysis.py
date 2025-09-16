# main_data_analysis.py

# --- Imports ---
from collections import namedtuple
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


from utils.labels_utils import get_label_pretty_name, get_sensor_names_from_features 
#utils.plotting has : figure_context_over_participation_time, figure_feature_scatter_for_labels, figure_feature_track_and_hist, 
# figure_pie_chart, jaccard_similarity_for_label_pairs
from utils.plotting import *
from utils.data_loading import read_user_data

# --- Constants and configurations ---
WORKING_DIR = os.getcwd()
WORKING_DIR_FILES = os.listdir(WORKING_DIR)
DATA_FOLDER = "ExtraSensory.per_uuid_features_labels"  # Unzipped data
if(DATA_FOLDER not in WORKING_DIR_FILES):
    print("The dataset folder was not found in the current directory.")
    exit(1)
DATA_FOLDER_PATH = os.path.join(WORKING_DIR, DATA_FOLDER)
RESULTS_FOLDER = "results\\data_analysis_per_user"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
USERS_DATA = os.listdir(DATA_FOLDER)
USERS_DATA_LIST = []
UserDataTuple = namedtuple("UserData", ["X", "Y", "M", "timestamps", "feature_names", "label_names"])

# Loop for all users to have their data in a list of tuples
# and create a folder for each user to save the results 
for uuid in USERS_DATA:
    user_filename = uuid.split(".")[0]
    user_results_folder = os.path.join(RESULTS_FOLDER, f"{user_filename}")
    os.makedirs(user_results_folder, exist_ok=True)
    user_outputs_path = os.path.join(user_results_folder, f"{user_filename}.txt")
    (X,Y,M,timestamps,feature_names,label_names) = read_user_data(DATA_FOLDER_PATH, uuid)
    USERS_DATA_LIST.append(UserDataTuple(X, Y, M, timestamps, feature_names, label_names))
    feat_sensor_names = get_sensor_names_from_features(feature_names)

    with open(user_outputs_path, "w") as file:
        sys.stdout = file

        print("The parts of the user's data (and their dimensions):")
        print("Every example has its timestamp, indicating the minute when the example was recorded")
        print("User %s has %d examples (~%d minutes of behavior)" % (uuid,len(timestamps),len(timestamps)))
        print(timestamps.shape)
        print("The primary data files have %d different sensor-features" % len(feature_names))
        print("X is the feature matrix. Each row is an example and each column is a sensor-feature:")
        print(X.shape)
        print("The primary data files have %s context-labels" % len(label_names))
        print("Y is the binary label-matrix. Each row represents an example and each column represents a label.")
        print("Value of 1 indicates the label is relevant for the example:")
        print(Y.shape)
        print("Y is accompanied by the missing-label-matrix, M.")
        print("Value of 1 indicates that it is best to consider an entry (example-label pair) as 'missing':")
        print(M.shape)

        print("")
        n_examples_per_label = np.sum(Y,axis=0)
        labels_and_counts = zip(label_names,n_examples_per_label)
        sorted_labels_and_counts = sorted(labels_and_counts,reverse=True,key=lambda pair:pair[1])
        print("How many examples does this user have for each context-label:")
        print("-"*20)
        for (label,count) in sorted_labels_and_counts:
            print("label %s - %d minutes" % (get_label_pretty_name(label),count))
            pass
        
        print("")
        for (fi,feature) in enumerate(feature_names):
            print("%3d) %s %s" % (fi,feat_sensor_names[fi].ljust(10),feature))
            pass

        sys.stdout = sys.__stdout__  # Restore after the for loop

    # Create the figure and axes for the pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), facecolor='white')

    labels_to_display = ['LYING_DOWN','SITTING','OR_standing','FIX_walking','FIX_running']
    figure_pie_chart(Y,label_names,labels_to_display,'Body state',ax1)

    labels_to_display = ['PHONE_IN_HAND','PHONE_IN_BAG','PHONE_IN_POCKET','PHONE_ON_TABLE']
    figure_pie_chart(Y,label_names,labels_to_display,'Phone position',ax2)

    # Save both subplots into one combined file
    fig_path = os.path.join(user_results_folder, "combined_pie_charts.png")
    save_and_close_figure(fig, fig_path)

    labels_to_display = ['LYING_DOWN','LOC_home','LOC_main_workplace','SITTING','OR_standing','FIX_walking',
                     'IN_A_CAR','ON_A_BUS','EATING']
    label_colors = ['g','y','b','c','m','b','r','k','purple']

    fig = figure_context_over_participation_time(timestamps,Y,label_names,labels_to_display,label_colors)
    fig_path = os.path.join(user_results_folder, "context_over_participation_time.png")
    save_and_close_figure(fig, fig_path)

    fig = figure_context_over_participation_time(timestamps,Y,label_names,labels_to_display,label_colors,use_actual_dates=True)
    fig_path = os.path.join(user_results_folder, "context_over_participation_time_with_dates.png")
    save_and_close_figure(fig, fig_path)

    J = jaccard_similarity_for_label_pairs(Y)
    
    fig = plt.figure(figsize=(10,10),facecolor='white')
    ax = plt.subplot(1,1,1)
    plt.imshow(J,interpolation='none') 
    plt.colorbar()

    pretty_label_names = [get_label_pretty_name(label) for label in label_names]
    n_labels = len(label_names)
    ax.set_xticks(range(n_labels))
    ax.set_xticklabels(pretty_label_names,rotation=45,ha='right',fontsize=7)
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels(pretty_label_names,fontsize=7)
    plt.title('Jaccard similarity for label pairs',fontsize=14)
    fig_path = os.path.join(user_results_folder, "jaccard_similarity.png") 
    save_and_close_figure(fig, fig_path)

    feature_inds = [0,102,133,148,157,158]
    fig_path = os.path.join(user_results_folder, "feature_track_and_hist_1_")
    figure_feature_track_and_hist(X,feature_names,timestamps,feature_inds, fig_path)

    feature_inds = [205,223]
    fig_path = os.path.join(user_results_folder, "feature_track_and_hist_2_")
    figure_feature_track_and_hist(X,feature_names,timestamps,feature_inds, fig_path)

    feature1 = 'proc_gyro:magnitude_stats:time_entropy' #raw_acc:magnitude_autocorrelation:period' 
    feature2 = 'raw_acc:3d:mean_y'
    label2color_map = {'PHONE_IN_HAND':'b','PHONE_ON_TABLE':'g'}
    fig = figure_feature_scatter_for_labels(X,Y,feature_names,label_names,feature1,feature2,label2color_map)
    fig_path = os.path.join(user_results_folder, "feature_scatter_for_labels.png")
    save_and_close_figure(fig, fig_path)
    
    feature1 = 'watch_acceleration:magnitude_spectrum:log_energy_band1'
    feature2 = 'watch_acceleration:3d:mean_z'
    label2color_map = {'FIX_walking':'b','WATCHING_TV':'g'}
    fig = figure_feature_scatter_for_labels(X,Y,feature_names,label_names,feature1,feature2,label2color_map)
    fig_path = os.path.join(user_results_folder, "feature_scatter_for_labels_2.png")
    save_and_close_figure(fig, fig_path)

    feature1 = 'proc_gyro:magnitude_stats:time_entropy' #raw_acc:magnitude_autocorrelation:period' 
    feature2 = 'raw_acc:3d:mean_y'
    label2color_map = {'PHONE_IN_HAND':'b','PHONE_ON_TABLE':'g'}
    fig = figure_feature_scatter_for_labels(X,Y,feature_names,label_names,feature1,feature2,label2color_map)
    fig_path = os.path.join(user_results_folder, "feature_scatter_for_phone_labels_1.png")
    save_and_close_figure(fig, fig_path)

    feature1 = 'watch_acceleration:magnitude_spectrum:log_energy_band1'
    feature2 = 'watch_acceleration:3d:mean_z'
    label2color_map = {'FIX_walking':'b','WATCHING_TV':'g'}
    fig = figure_feature_scatter_for_labels(X,Y,feature_names,label_names,feature1,feature2,label2color_map)
    fig_path = os.path.join(user_results_folder, "feature_scatter_for_phone_labels_2.png")
    save_and_close_figure(fig, fig_path)