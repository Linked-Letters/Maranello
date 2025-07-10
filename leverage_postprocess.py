#!/usr/bin/python
# This file is part of the Maranello software, a collection of python tools for
# processing and analyzing data from professional racing series.
# 
# Copyright (C) 2025 George Limpert
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>. 

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

def main ():
	# Get the parameters from the command line
	if len(sys.argv) < 2:
		print('Usage: '+sys.argv[0]+' <input file name> [advancement scale factor]')
		exit()
	input_file_name = sys.argv[1].strip()
	if len(sys.argv) >= 3:
		try:
			adv_scale = float(sys.argv[2].strip())
		except:
			print('Scale factor error')
			sys.exit(1)
	else:
		adv_scale = 1.0

	pickle_handle = open(input_file_name, 'rb')
	leverage_data = pickle.load(pickle_handle)
	pickle_handle.close()

	# Get the track names
	track_names = list(leverage_data['track_stats'].keys())

	# Scale the relevant statistics
	track_stat_types = ['leverage', 'advancement', 'excitement']
	track_stat_labels = ['Leverage', 'Advancement', 'Excitement']
	for track_name in track_names:
		for track_stat_type in track_stat_types:
			if ['advancement', 'excitement'].count(track_stat_type) > 0:
				leverage_data['track_stats'][track_name][track_stat_type] = np.multiply(leverage_data['track_stats'][track_name][track_stat_type], adv_scale)

	# Calculate mean/median/min/max of track statistics
	track_stats = {}
	for track_stat_type in track_stat_types:
		track_stats[track_stat_type] = {}
	for track_name in track_names:
		for track_stat_type in track_stat_types:
			track_stats[track_stat_type][track_name] = {}
			track_stats[track_stat_type][track_name]['start'] = leverage_data['track_stats'][track_name][track_stat_type][0]
			track_stats[track_stat_type][track_name]['finish'] = leverage_data['track_stats'][track_name][track_stat_type][-1]
			track_stats[track_stat_type][track_name]['min'] = np.amin(leverage_data['track_stats'][track_name][track_stat_type])
			track_stats[track_stat_type][track_name]['max'] = np.amax(leverage_data['track_stats'][track_name][track_stat_type])
			track_stats[track_stat_type][track_name]['mean'] = np.mean(leverage_data['track_stats'][track_name][track_stat_type])
			track_stats[track_stat_type][track_name]['median'] = np.median(leverage_data['track_stats'][track_name][track_stat_type])

	# Now, create plots for the three types of track statistics
	xvals = leverage_data['race_times_pct']
	track_stats_mean = {}
	for track_stat_id in range(0, len(track_stat_types), 1):
		track_stat_type = track_stat_types[track_stat_id]
		track_stat_name = track_stat_labels[track_stat_id]
		plt.figure(figsize = (6.5, 5.5), dpi = 150)
		plt.rcParams['font.family'] = 'Verdana'
		mpl.rcParams['text.antialiased'] = True
		plt.grid(True)
		cmap = mpl.colormaps['gist_rainbow']
		max_yvals = 0
		stat_mean = np.zeros(xvals.shape)
		for track_id in range(0, len(track_names), 1):
			track_name = track_names[track_id]
			yvals = leverage_data['track_stats'][track_name][track_stat_type]
			stat_mean = np.add(stat_mean, yvals)
			max_yvals = max(max_yvals, np.amax(yvals))
			plt.plot(xvals, yvals, linewidth = 0.7, color = cmap(track_id / (len(track_names) - 1)), label = track_name)
		stat_mean = np.divide(stat_mean, len(track_names))
		track_stats_mean[track_stat_type] = stat_mean
		plt.plot(xvals, stat_mean, linewidth = 1.0, color = (0.0, 0.0, 0.0, 1.0), label = 'Mean of Tracks')
		ax = plt.gca()
		ax.set_xlim([min(xvals), max(xvals)])
		ax.set_ylim([0.0, min(1.0, 1.1 * max_yvals)])
		ax.set_title(track_stat_name)
		ax.set_xlabel('Percentage of Race Completed')
		ax.tick_params(axis = 'x', labelrotation = 45)
		ax.legend(loc = 'center left', bbox_to_anchor = (1.02, 0.5), fancybox = True, prop = {'size': 6})
		plt.tight_layout()
		plt.savefig(track_stat_type + '.png', bbox_inches = 'tight', dpi = 150)
		plt.close()
		plt.clf()

	# To create a consistent presentation, get the maximum excitement and advancement
	max_exc_adv = 0.0
	for track_name in track_names:
		for track_stat_id in range(0, len(track_stat_types), 1):
			track_stat_type = track_stat_types[track_stat_id]
			if ['advancement', 'excitement'].count(track_stat_type) > 0:
				max_exc_adv = max(np.amax(leverage_data['track_stats'][track_name][track_stat_type]), max_exc_adv)

	# Now, create track plots
	for track_name in track_names:
		fig = plt.figure(figsize = (6.5, 5.5), dpi = 150)
		plt.rcParams['font.family'] = 'Verdana'
		mpl.rcParams['text.antialiased'] = True
		cmap = mpl.colormaps['gist_rainbow']
		max_yvals = 0
		ax = plt.gca()
		ax.set_xlim([min(xvals), max(xvals)])
		ax.set_ylim([0.0, 1.03 * max_exc_adv])
		ax.set_title(track_name)
		ax.set_xlabel('Percentage of Race Completed')
		ax.tick_params(axis = 'x', labelrotation = 45)		
		ax.set_ylabel('Advancement and Excitement')
		ax2 = ax.twinx()
		ax2.set_ylabel('Leverage')
		ax2.set_ylim([0.0, 1.0])
		for track_stat_id in range(0, len(track_stat_types), 1):
			track_stat_type = track_stat_types[track_stat_id]
			track_stat_name = track_stat_labels[track_stat_id]
			yvals = leverage_data['track_stats'][track_name][track_stat_type]
			max_yvals = max(max_yvals, np.amax(yvals))
			if track_stat_type == 'leverage':
				ax2.plot(xvals, yvals, linewidth = 2.5, color = cmap((track_stat_id * 0.8) / (len(track_stat_types) - 1) + 0.0), label = track_stat_name)
			else:
				ax.plot(xvals, yvals, linewidth = 2.5, color = cmap((track_stat_id * 0.8) / (len(track_stat_types) - 1) + 0.0), label = track_stat_name)
		# This is the mean of the data, which also gets plotted on this graph
		for track_stat_id in range(0, len(track_stat_types), 1):
			track_stat_type = track_stat_types[track_stat_id]
			track_stat_name = track_stat_labels[track_stat_id]
			yvals = track_stats_mean[track_stat_type]
			if track_stat_type == 'leverage':
				ax2.plot(xvals, yvals, linewidth = 1, color = tuple(np.divide(np.add(np.array(cmap((track_stat_id * 0.8) / (len(track_stat_types) - 1) + 0.0)), np.array([0.8, 0.8, 0.8, 1])), 2).tolist()), label = 'Mean ' + track_stat_name)
			else:
				ax.plot(xvals, yvals, linewidth = 1, color = tuple(np.divide(np.add(np.array(cmap((track_stat_id * 0.8) / (len(track_stat_types) - 1) + 0.0)), np.array([0.8, 0.8, 0.8, 1])), 2).tolist()), label = 'Mean ' + track_stat_name)
		fig.legend(loc = 'center left', bbox_to_anchor = (1.02, 0.5), fancybox = True, prop = {'size': 6})
		plt.tight_layout()
		plt.savefig('track_' + track_name + '.png', bbox_inches = 'tight', dpi = 150)
		plt.close()
		plt.clf()

	# Organize, sort, and plot data tables
	track_stat_columns = ['Circuit', 'Mean', 'Median', 'Maximum', 'Minimum', 'Start', 'Finish']
	for track_stat_id in range(0, len(track_stat_types), 1):
		track_stat_type = track_stat_types[track_stat_id]
		track_stat_name = track_stat_labels[track_stat_id]
		track_stat_list = []
		for track_name in track_names:
			track_stat_list.append([track_name, track_stats[track_stat_type][track_name]['mean'], track_stats[track_stat_type][track_name]['median'], track_stats[track_stat_type][track_name]['max'], track_stats[track_stat_type][track_name]['min'], track_stats[track_stat_type][track_name]['start'], track_stats[track_stat_type][track_name]['finish']])
		track_stat_list.append(['Mean of Tracks'] + np.mean(np.array([x[1:len(track_stat_columns)] for x in track_stat_list]), axis = 0).tolist())
		track_stat_sorted_list = sorted(track_stat_list, key = lambda x: x[1], reverse = True)

		# Plot the data
		output_plot_data_table = np.empty([len(track_stat_sorted_list) + 1, len(track_stat_columns)], dtype=object)
		output_plot_colors_table = np.empty([len(track_stat_sorted_list) + 1, len(track_stat_columns)], dtype=object)
		# Color code the data as appropriate
		for col_idx in range(0, len(track_stat_columns), 1):
			output_plot_data_table[0, col_idx] = track_stat_columns[col_idx]
			output_plot_colors_table[0, col_idx] = '#FF66FF'
			for row_idx in range(0, len(track_stat_sorted_list), 1):
				if col_idx == 0:
					output_plot_data_table[row_idx + 1, col_idx] = track_stat_sorted_list[row_idx][col_idx]
				else:
					output_plot_data_table[row_idx + 1, col_idx] = '{:.3f}'.format(track_stat_sorted_list[row_idx][col_idx])
				if track_stat_sorted_list[row_idx][0] == 'Mean of Tracks':
					output_plot_colors_table[row_idx + 1, :] = '#FFFF33'
				elif (row_idx % 2) == 0:
					output_plot_colors_table[row_idx + 1, :] = '#FFFFFF'
				else:
					output_plot_colors_table[row_idx + 1, :] = '#CCCCCC'
		output_plot_data_table[0, 0] = track_stat_name
		output_plot_colors_table[0, 0] = '#99FF99'
		# Now actually create the figure
		plt.figure(figsize = (6.5, 1), dpi = 150)
		ax = plt.subplot()
		ax.axis('off')
		plt.rcParams['font.family'] = 'Verdana'
		mpl.rcParams['text.antialiased'] = True
		table_colors = ax.table(cellText = output_plot_data_table, cellColours = output_plot_colors_table,  cellLoc = 'left', loc = 'center', edges = 'BLTR')
		for x in table_colors.properties()['celld'].values():
			x.set(linewidth=0)
		table_borders = ax.table(cellText = output_plot_data_table, cellLoc = 'left', loc = 'center', edges = 'BT')
		table_colors.auto_set_font_size(False)
		table_borders.auto_set_font_size(False)
		table_colors.auto_set_column_width(col = list(range(0, len(track_stat_columns), 1)))
		table_borders.auto_set_column_width(col = list(range(0, len(track_stat_columns), 1)))
		# Adjust the borders and the fonts as desired to make the table a bit easier to read
		for i in range(0, len(track_stat_sorted_list) + 1, 1):
			for j in range(0, len(track_stat_columns), 1):
				if i == 0:
					table_colors.get_celld()[(i, j)].set_text_props(fontweight = 'heavy', fontsize = 11)
					table_borders.get_celld()[(i, j)].set_text_props(fontweight = 'heavy', fontsize = 11)
				else:
					if j == 0:
						table_colors.get_celld()[(i, j)].set_text_props(fontweight = 'bold', fontsize = 10)
						table_borders.get_celld()[(i, j)].set_text_props(fontweight = 'bold', fontsize = 10)
					else:
						table_colors.get_celld()[(i, j)].set_text_props(fontsize = 10)
						table_borders.get_celld()[(i, j)].set_text_props(fontsize = 10)
					if output_plot_data_table[i, 0] == 'Mean of Tracks':
						table_colors.get_celld()[(i, j)].set_text_props(style = 'italic')
						table_borders.get_celld()[(i, j)].set_text_props(style = 'italic')
		plt.savefig('table_' + track_stat_type + '.png', bbox_inches = 'tight', dpi = 150)
		plt.close()
		plt.clf()

if __name__ == '__main__':
	main()


