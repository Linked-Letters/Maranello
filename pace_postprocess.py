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

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats as stats
import sys

# How far on either side of the race do we average
average_half_window = 3

# How large is the confidence interval that we calculate
ci_interval_size = 0.8

# These are team colors
team_color_lookup_seasonal = {
	2025: {
		'Alpine': (0, 147, 204),
		'Kick Sauber': (82, 226, 82),
		'Williams': (100, 196, 255),
		'Aston Martin': (34, 153, 113),
		'Red Bull Racing': (54, 113, 198),
		'Mercedes': (39, 244, 210),
		'Haas F1 Team': (182, 186, 189),
		'Ferrari': (232, 0, 32),
		'McLaren': (255, 128, 0),
		'Racing Bulls': (100, 146, 255)
	},
	2024: {
		'Alpine': (255, 135, 188),
		'Kick Sauber': (82, 226, 82),
		'Williams': (100, 196, 255),
		'Aston Martin': (34, 153, 113),
		'Red Bull Racing': (54, 113, 198),
		'Mercedes': (39, 244, 210),
		'Haas F1 Team': (182, 186, 189),
		'Ferrari': (232, 0, 45),
		'McLaren': (255, 128, 0),
		'RB': (100, 146, 255)
	},
	2023: {
		'Alpine': (34, 147, 209),
		'Alfa Romeo': (201, 45, 75),
		'Williams': (55, 190, 221),
		'Aston Martin': (53, 140, 117),
		'Red Bull Racing': (54, 113, 198),
		'Mercedes': (108, 211, 191),
		'Haas F1 Team': (182, 186, 189),
		'Ferrari': (249, 21, 54),
		'McLaren': (245, 128, 32),
		'AlphaTauri': (94, 143, 170)
	},
	2022: {
		'Alpine': (34, 147, 209),
		'Alfa Romeo': (172, 32, 57),
		'Williams': (55, 190, 221),
		'Aston Martin': (45, 130, 109),
		'Red Bull Racing': (30, 91, 198),
		'Mercedes': (108, 211, 191),
		'Haas F1 Team': (182, 186, 189),
		'Ferrari': (237, 28, 46),
		'McLaren': (245, 128, 32),
		'AlphaTauri': (78, 124, 155)
	},
	2021: {
		'Alpine': (0, 144, 255),
		'Alfa Romeo Racing': (144, 0, 0),
		'Williams': (0, 90, 255),
		'Aston Martin': (0, 111, 98),
		'Red Bull Racing': (6, 0, 239),
		'Mercedes': (0, 210, 190),
		'Haas F1 Team': (255, 255, 255),
		'Ferrari': (220, 0, 0),
		'McLaren': (255, 135, 0),
		'AlphaTauri': (43, 69, 98)
	},
	2020: {
		'Renault': (255, 245, 0),
		'Alfa Romeo Racing': (150, 0, 0),
		'Williams': (0, 130, 250),
		'Racing Point': (245, 150, 200),
		'Red Bull Racing': (6, 0, 239),
		'Mercedes': (0, 210, 190),
		'Haas F1 Team': (120, 120, 120),
		'Ferrari': (192, 0, 0),
		'McLaren': (255, 135, 0),
		'AlphaTauri': (200, 200, 200)
	},
	2019: {
		'Renault': (255, 245, 0),
		'Alfa Romeo Racing': (155, 0, 0),
		'Williams': (255, 255, 255),
		'Racing Point': (245, 150, 200),
		'Red Bull Racing': (30, 65, 255),
		'Mercedes': (0, 210, 190),
		'Haas F1 Team': (240, 215, 135),
		'Ferrari': (220, 0, 0),
		'McLaren': (255, 135, 0),
		'Toro Rosso': (70, 155, 255)
	},
	2018: {
		'Renault': (255, 245, 0),
		'Sauber': (155, 0, 0),
		'Williams': (255, 255, 255),
		'Force India': (245, 150, 200),
		'Racing Point': (245, 150, 200),
		'Red Bull Racing': (0, 50, 125),
		'Mercedes': (0, 210, 190),
		'Haas F1 Team': (90, 90, 90),
		'Ferrari': (220, 0, 0),
		'McLaren': (255, 135, 0),
		'Toro Rosso': (0, 50, 255)
	}
}

def main ():
	global average_half_window, ci_interval_size, team_color_lookup_seasonal

	# Get the parameters from the command line
	if len(sys.argv) < 3:
		print('Usage: '+sys.argv[0]+' <input file name> <output image>')
		exit()
	input_file_name = sys.argv[1].strip()
	output_file_name = sys.argv[2].strip()

	pickle_handle = open(input_file_name, 'rb')
	performance_data = pickle.load(pickle_handle)
	pickle_handle.close()

	season = performance_data['season']

	# Look up the colors for the season
	if list(team_color_lookup_seasonal.keys()).count(season) > 0:
		team_color_lookup = team_color_lookup_seasonal[season]
	else:
		team_color_lookup = {}

	# Set up some basic data structures, including a list of the races (ordered by the weekend number)
	race_keys = sorted(list(performance_data['races'].keys()))
	team_names = []
	weekend_performance = {}

	# Loop through the races
	for race_idx in range(0, len(race_keys), 1):
		race_idx_min = max(0, race_idx - average_half_window)
		race_idx_max = min(len(race_keys) - 1, race_idx + average_half_window)

		race_id = race_keys[race_idx]
		weekend_performance[race_id] = {}
		race_performance_data = {}
		# The analysis includes not only the current race, but a few races before or after it, so retrieve that data and merge it
		for iter_race_id in race_keys[race_idx_min:race_idx_max + 1]:
			race_reference_lap = performance_data['races'][iter_race_id]['reference_lap']
			# Loop through each team for which we have data
			for team_name in list(performance_data['races'][iter_race_id]['team_data'].keys()):
				if list(race_performance_data.keys()).count(team_name) == 0:
					race_performance_data[team_name] = {}
					race_performance_data[team_name]['actual'] = []
					race_performance_data[team_name]['predicted'] = []
				# This adjusts the data so that the baseline is a 100 second lap, then adds it to the arrays
				race_performance_data[team_name]['actual'].extend(np.multiply(performance_data['races'][iter_race_id]['team_data'][team_name]['actual'], 100 / race_reference_lap).tolist())
				race_performance_data[team_name]['predicted'].extend(np.multiply(performance_data['races'][iter_race_id]['team_data'][team_name]['predicted'], 100 / race_reference_lap).tolist())
		# Loop through each team in the data and calculate some statistics on their performance
		for team_name in list(race_performance_data.keys()):
			team_names.append(team_name)
			team_pace_lap_difference = np.subtract(np.array(race_performance_data[team_name]['actual']), np.array(race_performance_data[team_name]['predicted']))
			team_pace_difference_mean = np.mean(team_pace_lap_difference)
			team_pace_difference_stdev = np.std(team_pace_lap_difference)
			team_pace_difference_n = len(race_performance_data[team_name]['actual'])
			team_pace_difference_sem = stats.sem(team_pace_lap_difference)
			team_pace_difference_ci_width = team_pace_difference_sem * stats.t.ppf((1 + ci_interval_size) / 2, team_pace_difference_n - 1)
			weekend_performance[race_id][team_name] = {}
			weekend_performance[race_id][team_name]['mean'] = team_pace_difference_mean
			weekend_performance[race_id][team_name]['stdev'] = team_pace_difference_stdev
			weekend_performance[race_id][team_name]['n'] = team_pace_difference_n
			weekend_performance[race_id][team_name]['sem'] = team_pace_difference_sem
			weekend_performance[race_id][team_name]['ci_width'] = team_pace_difference_ci_width

	# Find the ordering of the teams at the last week of the data set, allowing the data to be sorted accordingly; this is important so the ordering in the legend is consistent with the most recent week, and makes it easier to figure out which line corresponds to a team
	team_names = list(set(team_names))
	team_sorted_names = []
	for team_name in team_names:
		team_row = [team_name, 0.0]
		for race_id in race_keys:
			if list(weekend_performance[race_id].keys()).count(team_name) > 0:
				team_row = [team_name, weekend_performance[race_id][team_name]['mean']]
		team_sorted_names.append(team_row)
	team_sorted_names = sorted(team_sorted_names, key = lambda x: x[1], reverse = False)
	team_sorted_names = [x[0] for x in team_sorted_names]

	# Create the figure to show the data, with a black background to better show the typical F1 team colors
	fig = plt.figure(figsize = (7.5, 5.5), dpi = 150)
	plt.rcParams['font.family'] = 'Verdana'
	mpl.rcParams['text.antialiased'] = True
	ax = plt.gca()
	ax.set_facecolor((0.0, 0.0, 0.0))
	fig.set_facecolor((0.0, 0.0, 0.0))
	ymax = 0
	# Loop through each team in the data and plot a background fill as well as an outline around an actual team's pace, showing both the uncertainty and trying to make the actual line for the team's performance a bit easier to see
	for team_name in team_sorted_names:
		x_vals = []
		y_vals = []
		y_min_vals = []
		y_max_vals = []
		# Compile the team's data from the race-by-race data
		for race_id in race_keys:
			if list(weekend_performance[race_id].keys()).count(team_name) > 0:
				x_vals.append(race_id)
				y_vals.append(weekend_performance[race_id][team_name]['mean'])
				y_min_vals.append(weekend_performance[race_id][team_name]['mean'] - weekend_performance[race_id][team_name]['ci_width'])
				y_max_vals.append(weekend_performance[race_id][team_name]['mean'] + weekend_performance[race_id][team_name]['ci_width'])
				ymax = max(ymax, max(abs(weekend_performance[race_id][team_name]['mean'] - weekend_performance[race_id][team_name]['ci_width']), abs(weekend_performance[race_id][team_name]['mean'] + weekend_performance[race_id][team_name]['ci_width'])))
		team_color = tuple([x / 255.0 for x in team_color_lookup[team_name]])
		# Add a background outline that's bright to try to stand out against the color fills
		team_color_hsv = mpl.colors.rgb_to_hsv(team_color)
		team_color_hsv[2] = (2.0 + team_color_hsv[2]) / 3
		team_outline_rgb = mpl.colors.hsv_to_rgb(team_color_hsv)
		ax.plot(x_vals, y_vals, linewidth = 2, color = team_outline_rgb, alpha = 0.6)
		# Adjust the color fills to try to make them as visible as possible, too
		team_color_hsv = mpl.colors.rgb_to_hsv(team_color)
		team_color_hsv[1] = min(team_color_hsv[1] * 1.25, 1.0)
		team_color_hsv[2] = 0.4
		team_fill_rgb = mpl.colors.hsv_to_rgb(team_color_hsv)
		ax.fill_between(x_vals, y_min_vals, y_max_vals, alpha = 0.5, linewidth = 0, color = team_fill_rgb)
	# Now, plot a thin solid line with the pace for each team
	for team_name in team_sorted_names:
		x_vals = []
		y_vals = []
		# Compile the team's data from the race-by-race data and plot it
		for race_id in race_keys:
			if list(weekend_performance[race_id].keys()).count(team_name) > 0:
				x_vals.append(race_id)
				y_vals.append(weekend_performance[race_id][team_name]['mean'])
		team_color = tuple([x / 255.0 for x in team_color_lookup[team_name]])
		ax.plot(x_vals, y_vals, linewidth = 1, color = team_color, label = team_name)
	# Configure the axes and add a grid
	ax.yaxis.set_inverted(True)
	ax.set_ylim([1.02 * ymax, -1.02 * ymax])
	ax.set_xlim([min(race_keys), max(race_keys)])
	ax.set_xticks(ticks = x_vals, minor = False)
	ax.set_xlabel('Round')
	ax.set_ylabel('Pace difference (seconds) for 100 second lap')
	ax.set_title('Team pace during the ' + '{:d}'.format(season) + ' season')
	ax.grid(True, which = 'both')
	# Add a legend and adjust the line thicknesses to make them look like what actually appears on the line graph
	lgnd = fig.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5), fancybox = True, prop = {'size': 8}, labelcolor = (1.0, 1.0, 1.0), facecolor = (0.0, 0.0, 0.0))
	for lgnd_hdl in lgnd.legend_handles:
		lgnd_hdl.set_linewidth(1.5)
	# Adjust the colors of the axes and any lines to make them visible against the background
	ax.xaxis.label.set_color((1.0, 1.0, 1.0))
	ax.yaxis.label.set_color((1.0, 1.0, 1.0))
	ax.title.set_color((1.0, 1.0, 1.0))
	[ax.spines[x].set_color((1.0, 1.0, 1.0)) for x in ax.spines]
	[x.set_color((0.6, 0.6, 0.6)) for x in ax.get_xgridlines()]
	[x.set_color((0.6, 0.6, 0.6)) for x in ax.get_ygridlines()]
	ax.xaxis.set_tick_params(colors = (1.0, 1.0, 1.0), which = 'both')
	ax.yaxis.set_tick_params(colors = (1.0, 1.0, 1.0), which = 'both')
	[x.set_color((1.0, 1.0, 1.0)) for x in ax.xaxis.get_ticklabels()]
	[x.set_color((1.0, 1.0, 1.0)) for x in ax.yaxis.get_ticklabels()]
	plt.tight_layout()
	plt.savefig(output_file_name, bbox_inches = 'tight', dpi = 150)
	plt.close()
	plt.clf()

if __name__ == '__main__':
	main()
