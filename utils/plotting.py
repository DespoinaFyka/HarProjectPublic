import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
from matplotlib import pyplot as plt
import numpy as np
import datetime
import pytz

from utils.labels_utils import get_label_pretty_name


# --- Plotting functions ---
def save_and_close_figure(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory

# Plotting function for pie charts
# This function creates a pie chart for the given labels and their corresponding portions of time.
def figure_pie_chart(Y, label_names, labels_to_display, title_str, ax):
    n_samples = Y.shape[0]
    
    # Create mapping from label to index
    label_to_index = {label: i for i, label in enumerate(label_names)}

    # Initialize portions and labels
    portions_to_display = []
    pretty_labels_to_display = []
    zero_labels = []

    total_mask = np.zeros(n_samples, dtype=bool)  # for tracking labeled entries

    for label in labels_to_display:
        if label not in label_to_index:
            continue

        idx = label_to_index[label]
        label_column = Y[:, idx]

        # Count how many samples have a 1 for this label
        portion = np.nansum(label_column) / n_samples
        if portion > 0:
            portions_to_display.append(portion)
            pretty_labels_to_display.append(get_label_pretty_name(label))
        else:
            zero_labels.append(get_label_pretty_name(label))

        # Update mask to count which samples had any label assigned
        total_mask |= ~np.isnan(label_column)

    # Add MISSING portion
    portion_missing = 1.0 - np.sum(portions_to_display)
    if portion_missing > 0:
        portions_to_display.append(portion_missing)
        pretty_labels_to_display.append("MISSING")

    # Plot
    ax.pie(portions_to_display, labels=pretty_labels_to_display, autopct='%.2f%%')
    ax.axis('equal')
    ax.set_title(title_str)

    # Add note about excluded (zero) labels
    if zero_labels:
        zero_note = "Excluded (0%): " + ", ".join(zero_labels)
        ax.text(0, -1.2, zero_note, ha='center', fontsize=8, color='gray')

def get_actual_date_labels(tick_seconds):
    time_zone = pytz.timezone('US/Pacific') # Assuming the data comes from PST time zone
    weekday_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    datetime_labels = []
    for timestamp in tick_seconds:
        tick_datetime = datetime.datetime.fromtimestamp(timestamp,tz=time_zone)
        weekday_str = weekday_names[tick_datetime.weekday()]
        time_of_day = tick_datetime.strftime('%I:%M%p')
        datetime_labels.append('%s\n%s' % (weekday_str,time_of_day))
        pass
    
    return datetime_labels
    
def figure_context_over_participation_time(timestamps,Y,label_names,labels_to_display,label_colors,use_actual_dates=False):
    fig = plt.figure(figsize=(10,7),facecolor='white')
    ax = plt.subplot(1,1,1)
    
    seconds_in_day = (60*60*24)

    ylabels = []
    ax.plot(timestamps,len(ylabels)*np.ones(len(timestamps)),'|',color='0.5',label='(Collected data)')
    ylabels.append('(Collected data)')

    for (li,label) in enumerate(labels_to_display):
        lind = label_names.index(label)
        is_label_on = Y[:,lind]
        label_times = timestamps[is_label_on]

        label_str = get_label_pretty_name(label)
        ax.plot(label_times,len(ylabels)*np.ones(len(label_times)),'|',color=label_colors[li],label=label_str)
        ylabels.append(label_str)
        pass

    tick_seconds = range(timestamps[0],timestamps[-1],seconds_in_day)
    if use_actual_dates:
        tick_labels = get_actual_date_labels(tick_seconds)
        plt.xlabel('Time in San Diego',fontsize=14)
        pass
    else:
        tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(seconds_in_day)).astype(int)
        plt.xlabel('days of participation',fontsize=14)
        pass
    
    ax.set_xticks(tick_seconds)
    ax.set_xticklabels(tick_labels,fontsize=14)

    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels,fontsize=14)

    ax.set_ylim([-1,len(ylabels)])
    ax.set_xlim([min(timestamps),max(timestamps)])
    return fig

def jaccard_similarity_for_label_pairs(Y):
    (n_examples,n_labels) = Y.shape
    Y = Y.astype(int)
    # For each label pair, count cases of:
    # Intersection (co-occurrences) - cases when both labels apply:
    both_labels_counts = np.dot(Y.T,Y)
    # Cases where neither of the two labels applies:
    neither_label_counts = np.dot((1-Y).T,(1-Y))
    # Union - cases where either of the two labels (or both) applies (this is complement of the 'neither' cases):
    either_label_counts = n_examples - neither_label_counts
    # Jaccard similarity - intersection over union:
    J = np.where(either_label_counts > 0, both_labels_counts.astype(float) / either_label_counts, 0.)
    return J

def figure_feature_track_and_hist(X,feature_names,timestamps,feature_inds, fig_path):
    seconds_in_day = (60*60*24)
    days_since_participation = (timestamps - timestamps[0]) / float(seconds_in_day)
    
    i = 1
    for ind in feature_inds:
        feature = feature_names[ind]
        feat_values = X[:, ind]

        fig = plt.figure(figsize=(10,3),facecolor='white')

        # --- Plot 1 (function of time) ---
        ax1 = plt.subplot(1,2,1)
        ax1.plot(days_since_participation,feat_values,'.-',markersize=3,linewidth=0.1)
        plt.xlabel('days of participation')
        plt.ylabel('feature value')
        plt.title('%d) %s\nfunction of time' % (ind,feature))

        # --- Plot 2 (histogram) ---
        ax1 = plt.subplot(1,2,2)
        existing_feature = np.logical_not(np.isnan(feat_values))
        ax1.hist(feat_values[existing_feature],bins=30)
        plt.xlabel('feature value')
        plt.ylabel('count')
        plt.title('%d) %s\nhistogram' % (ind,feature))

        fig.savefig(f"{fig_path}{i}.png", dpi=300, bbox_inches='tight');
        plt.close(fig)  # Close figure to free memory
        i += 1
    
    return

def figure_feature_scatter_for_labels(X,Y,feature_names,label_names,feature1,feature2,label2color_map):
    feat_ind1 = feature_names.index(feature1)
    feat_ind2 = feature_names.index(feature2)
    example_has_feature1 = np.logical_not(np.isnan(X[:,feat_ind1]))
    example_has_feature2 = np.logical_not(np.isnan(X[:,feat_ind2]))
    example_has_features12 = np.logical_and(example_has_feature1,example_has_feature2)
    
    fig = plt.figure(figsize=(12,5),facecolor='white')
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,4)
    
    for label in label2color_map.keys():
        label_ind = label_names.index(label)
        pretty_name = get_label_pretty_name(label)
        color = label2color_map[label]
        style = '.%s' % color
        
        is_relevant_example = np.logical_and(example_has_features12,Y[:,label_ind])
        count = sum(is_relevant_example)
        feat1_vals = X[is_relevant_example,feat_ind1]
        feat2_vals = X[is_relevant_example,feat_ind2]
        ax1.plot(feat1_vals,feat2_vals,style,markersize=5,label=pretty_name)
        
        #ax2.hist(X[is_relevant_example,feat_ind1],bins=20,normed=True,color=color,alpha=0.5,label='%s (%d)' % (pretty_name,count))
        ax2.hist(X[is_relevant_example,feat_ind1], bins=20, density=True, color=color, alpha=0.5, label='%s (%d)' % (pretty_name, count))
        #ax3.hist(X[is_relevant_example,feat_ind2],bins=20,normed=True,color=color,alpha=0.5,label='%s (%d)' % (pretty_name,count))
        ax3.hist(X[is_relevant_example,feat_ind2], bins=20, density=True, color=color, alpha=0.5, label='%s (%d)' % (pretty_name, count))
        pass
    
    ax1.set_xlabel(feature1)
    ax1.set_ylabel(feature2)
    
    ax2.set_title(feature1)
    ax3.set_title(feature2)
    
    ax2.legend(loc='best')

    fig.tight_layout()
    return fig