import numpy as np
import pandas as pd 
import operator
import os
import random

from sklearn.cluster import KMeans
from scipy.signal import find_peaks_cwt
from matplotlib import pyplot as plt

def read_file(filename):
	with open(filename, 'r') as f:
		lines = [line.split() for line in f.readlines()]
		all_points = []
		for line in lines:
			for v in line:
				all_points += [float(v)]

		labeled_points = {}
		for i in range(0, len(all_points), 66):
			joint = 0
			
			for j in range(i,i+66,3):
				if joint not in labeled_points:
					labeled_points[joint] = []

				temp_dict = {}
				temp_dict['x'] = all_points[j]
				temp_dict['y'] = all_points[j+1]
				temp_dict['z'] = all_points[j+2]
				labeled_points[joint] += [temp_dict]
				joint += 1

		return labeled_points

def find_reps(df_dominant_og, plot=False):
	
	if plot:

		plt.plot([i for i in range(0, len(df_dominant_og))], df_dominant_og*-1)
		plt.show()

	# Normalize by mean and multiply by -1 for squat to get more accurate peaks
	df_dominant = (df_dominant_og - np.mean(df_dominant_og))*-1

	max_signal = np.amax(df_dominant)
	min_signal = np.amin(df_dominant)

	# input data is of 10 repetitons
	n_rep = 10

	max_threshold = 0.4 * max_signal
	min_threshold = 0.4 * min_signal
	X = np.where( df_dominant > max_threshold)[0].reshape(-1, 1)

	# Finding k-means of peaks 
	try:
		kmeans = KMeans(n_clusters=n_rep+1, random_state=0).fit(X)
		max_peak_ind = np.sort( kmeans.cluster_centers_.reshape((1,-1))).astype(int)

		X = np.where( df_dominant < min_threshold)[0].reshape(-1, 1)
		kmeans = KMeans(n_clusters=n_rep+1, random_state=0).fit(X)
		min_peak_ind = np.sort( kmeans.cluster_centers_.reshape((1,-1))).astype(int)
	except ValueError:
		return df_dominant_og, 0


	# Counting repetitions in data
	current_rep = 0
	current_state = 1
	repetition_count = []
	state = [] #Up:1 down:-1
	time_in_current_state = 0
	time_threshold = 5
	for idx,s in enumerate(df_dominant):
	    if idx in max_peak_ind and current_state == 1 and time_in_current_state > time_threshold:
	        current_rep +=1
	        current_state = -1
	        time_in_current_state = 0
	        #print (idx)
	    elif idx in min_peak_ind and current_state == -1 and time_in_current_state > time_threshold:
	        current_state = 1
	        time_in_current_state = 0
	    else: 
	        time_in_current_state +=1
	    repetition_count.append(current_rep)
	    state.append(current_state)

	return df_dominant_og, repetition_count[-1]

def augment_data(movement_num, joint, axis):
	# Joint index in paper starts with 1
	joint = joint - 1

	df = pd.DataFrame()
	print('Movement '+ str(movement_num))
	results = {}
	if movement_num == 10:
		m_str = 'm10'
	else:
		m_str = 'm0' + str(movement_num)
	reps = []

	# Traverse over all files for this movement in directory
	for f in os.listdir(os.getcwd()):
		if f.endswith('.txt') and f.startswith(m_str):
			print('traversing', f)
			parsed_file = read_file(f)
			
			# Get the appropriate joint and axis 
			df_dominant_og = pd.DataFrame(parsed_file[joint])
			df_dominant_og = df_dominant_og[axis]

			# Get the number of reps 
			row, temp_reps = find_reps(df_dominant_og)
			reps += [temp_reps]
			df = df.append(row)

	df_to_write = df.copy(deep=True)
	for i in range(0,30):
		print('iteration',i)

		# Iterate over all clean row data
		for counter, row in df.iterrows():

			# Add noise with std deviation 1 
			noise1 = np.random.normal(0, 1, len(row))
			df_to_write = df_to_write.append(row + noise1)

			# Add noise with std deviation 5
			noise5 = np.random.normal(0, 5, len(row))
			df_to_write = df_to_write.append(row + noise5)

			# stretch signal
			df_to_write = df_to_write.append(row*1.5)

			row_clean = np.trim_zeros(row)
			row_len = len(row_clean)
			
			# Pick random indices to truncate rows
			start_i = random.randint(0, 3*(row_len//10))
			end_i = start_i + (random.randint(2,6)*row_len)//10
			new_df = row_clean[start_i:end_i]

			# find peaks in truncated rows 
			_, temp_reps = find_reps(new_df)
			new_df = new_df.reset_index(drop=True)
			df_to_write = df_to_write.append(new_df)

			# add noise with std deviation 1 to truncated row 
			noise1 = np.random.normal(0, 1, len(new_df))
			df_to_write = df_to_write.append(new_df + noise1)

			# add noise with std deviation 5 to trncated row 
			noise5 = np.random.normal(0, 5, len(new_df))
			df_to_write = df_to_write.append(new_df + noise5)

			# stretch truncated row 
			df_to_write = df_to_write.append(new_df * 1.5)

			# add # of reps to vector			
			reps += [temp_reps]*7

	df_to_write['reps'] = reps
	print('df shape', df_to_write.shape)
	df_to_write.to_csv('clean_and_gaussian.csv')
	print('wrote to csv')

augment_data(1, 0, 'y')

