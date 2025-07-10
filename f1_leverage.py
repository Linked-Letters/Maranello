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

import fastf1
import numpy as np
import pickle
import scipy.stats as stats
import sys

# Frequency (how often through the race do we determine the stats) and interval of race calculations (the window over which the data are averaged when doing the calculations)
calc_frequency = 0.01
calc_interval = 0.1

# Do we include a separate entry with all tracks over the period?
include_all_tracks = False

# If we're including all tracks, how many races are in each year?
year_race_count = {2018: 21, 2019: 21, 2020: 17, 2021: 22, 2022: 22, 2023: 22, 2024: 24}

# And if we're including all tracks, here's a list of races to exclude, with each excluded race having the year and race number in a list (e.g., [2021, 12] is the 2021 Belgian Grand Prix, which is widely considered a farce)
exclusion_list = [[2021, 12]]

# These are a list of races at each track, leaving out Belgium 2021 because it's just not representative of the racing at that circuit
track_list = {
	'Monaco': {'races': [{'year': 2018, 'weekend': 6}, {'year': 2019, 'weekend': 6}, {'year': 2021, 'weekend': 5}, {'year': 2022, 'weekend': 7}, {'year': 2023, 'weekend': 6}, {'year': 2024, 'weekend': 8}], 'type': 'street course'},
	'Singapore': {'races': [{'year': 2018, 'weekend': 15}, {'year': 2019, 'weekend': 15}, {'year': 2022, 'weekend': 17}, {'year': 2023, 'weekend': 15}, {'year': 2024, 'weekend': 18}], 'type': 'street course'},
	'Great Britain': {'races': [{'year': 2018, 'weekend': 10}, {'year': 2019, 'weekend': 10}, {'year': 2020, 'weekend': 4}, {'year': 2020, 'weekend': 5}, {'year': 2021, 'weekend': 10}, {'year': 2022, 'weekend': 10}, {'year': 2023, 'weekend': 10}, {'year': 2024, 'weekend': 12}], 'type': 'road course'},
	'Brazil': {'races': [{'year': 2018, 'weekend': 20}, {'year': 2019, 'weekend': 20}, {'year': 2021, 'weekend': 19}, {'year': 2022, 'weekend': 21}, {'year': 2023, 'weekend': 20}, {'year': 2024, 'weekend': 21}], 'type': 'road course'},
	'Bahrain': {'races': [{'year': 2018, 'weekend': 2}, {'year': 2019, 'weekend': 2}, {'year': 2020, 'weekend': 15}, {'year': 2021, 'weekend': 1}, {'year': 2022, 'weekend': 1}, {'year': 2023, 'weekend': 1}, {'year': 2024, 'weekend': 1}], 'type': 'road course'},
	'United States': {'races': [{'year': 2018, 'weekend': 18}, {'year': 2019, 'weekend': 19}, {'year': 2021, 'weekend': 17}, {'year': 2022, 'weekend': 19}, {'year': 2023, 'weekend': 18}, {'year': 2024, 'weekend': 19}], 'type': 'road course'},
	'Azerbaijan': {'races': [{'year': 2018, 'weekend': 4}, {'year': 2019, 'weekend': 4}, {'year': 2021, 'weekend': 6}, {'year': 2022, 'weekend': 8}, {'year': 2023, 'weekend': 4}, {'year': 2024, 'weekend': 17}], 'type': 'street course'},
	'Japan': {'races': [{'year': 2018, 'weekend': 17}, {'year': 2019, 'weekend': 17}, {'year': 2022, 'weekend': 18}, {'year': 2023, 'weekend': 16}, {'year': 2024, 'weekend': 4}], 'type': 'road course'},
	'Austria': {'races': [{'year': 2018, 'weekend': 9}, {'year': 2019, 'weekend': 9}, {'year': 2020, 'weekend': 1}, {'year': 2020, 'weekend': 2}, {'year': 2021, 'weekend': 8}, {'year': 2021, 'weekend': 9}, {'year': 2022, 'weekend': 11}, {'year': 2023, 'weekend': 9}, {'year': 2024, 'weekend': 11}], 'type': 'road course'},
	'Spain': {'races': [{'year': 2018, 'weekend': 5}, {'year': 2019, 'weekend': 5}, {'year': 2020, 'weekend': 6}, {'year': 2021, 'weekend': 4}, {'year': 2022, 'weekend': 6}, {'year': 2023, 'weekend': 7}, {'year': 2024, 'weekend': 10}], 'type': 'road course'},
	'Italy': {'races': [{'year': 2018, 'weekend': 14}, {'year': 2019, 'weekend': 14}, {'year': 2020, 'weekend': 8}, {'year': 2021, 'weekend': 14}, {'year': 2022, 'weekend': 16}, {'year': 2023, 'weekend': 14}, {'year': 2024, 'weekend': 16}], 'type': 'road course'},
	'Belgium': {'races': [{'year': 2018, 'weekend': 13}, {'year': 2019, 'weekend': 13}, {'year': 2020, 'weekend': 7}, {'year': 2022, 'weekend': 14}, {'year': 2023, 'weekend': 12}, {'year': 2024, 'weekend': 14}], 'type': 'road course'},
	'Hungary': {'races': [{'year': 2018, 'weekend': 12}, {'year': 2019, 'weekend': 12}, {'year': 2020, 'weekend': 3}, {'year': 2021, 'weekend': 11}, {'year': 2022, 'weekend': 13}, {'year': 2023, 'weekend': 11}, {'year': 2024, 'weekend': 13}], 'type': 'road course'},
	'Canada': {'races': [{'year': 2018, 'weekend': 7}, {'year': 2019, 'weekend': 7}, {'year': 2022, 'weekend': 9}, {'year': 2023, 'weekend': 8}, {'year': 2024, 'weekend': 9}], 'type': 'street course'},
	'Abu Dhabi': {'races': [{'year': 2018, 'weekend': 21}, {'year': 2019, 'weekend': 21}, {'year': 2020, 'weekend': 17}, {'year': 2021, 'weekend': 22}, {'year': 2022, 'weekend': 22}, {'year': 2023, 'weekend': 22}, {'year': 2024, 'weekend': 24}], 'type': 'road course'},
	'Australia': {'races': [{'year': 2018, 'weekend': 1}, {'year': 2019, 'weekend': 1}, {'year': 2022, 'weekend': 3}, {'year': 2023, 'weekend': 3}, {'year': 2024, 'weekend': 3}], 'type': 'street course'},
	'France': {'races': [{'year': 2018, 'weekend': 8}, {'year': 2019, 'weekend': 8}, {'year': 2021, 'weekend': 7}, {'year': 2022, 'weekend': 12}], 'type': 'road course'},
	'Netherlands': {'races': [{'year': 2021, 'weekend': 13}, {'year': 2022, 'weekend': 15}, {'year': 2023, 'weekend': 13}, {'year': 2024, 'weekend': 15}], 'type': 'road course'},
	'Mexico': {'races': [{'year': 2018, 'weekend': 19}, {'year': 2019, 'weekend': 18}, {'year': 2021, 'weekend': 18}, {'year': 2022, 'weekend': 20}, {'year': 2023, 'weekend': 19}, {'year': 2024, 'weekend': 20}], 'type': 'road course'},
	'Saudi Arabia': {'races': [{'year': 2021, 'weekend': 21}, {'year': 2022, 'weekend': 2}, {'year': 2023, 'weekend': 2}, {'year': 2024, 'weekend': 2}], 'type': 'street course'},
	'Imola': {'races': [{'year': 2020, 'weekend': 13}, {'year': 2021, 'weekend': 2}, {'year': 2022, 'weekend': 4}, {'year': 2024, 'weekend': 7}], 'type': 'road course'},
	'Qatar': {'races': [{'year': 2021, 'weekend': 20}, {'year': 2023, 'weekend': 17}, {'year': 2024, 'weekend': 23}], 'type': 'road course'},
	'China': {'races': [{'year': 2018, 'weekend': 3}, {'year': 2019, 'weekend': 3}, {'year': 2024, 'weekend': 5}], 'type': 'road course'},
	'Russia': {'races': [{'year': 2018, 'weekend': 16}, {'year': 2019, 'weekend': 16}, {'year': 2020, 'weekend': 10}, {'year': 2021, 'weekend': 15}], 'type': 'street course'},
	'Miami': {'races': [{'year': 2022, 'weekend': 5}, {'year': 2023, 'weekend': 5}, {'year': 2024, 'weekend': 6}], 'type': 'street course'}
}

def main ():
	global calc_frequency, calc_interval, include_all_tracks, year_race_count, exclusion_list, track_list

	# Get the parameters from the command line
	if len(sys.argv) < 2:
		print('Usage: '+sys.argv[0]+' <output file name>')
		exit()
	output_file_name = sys.argv[1].strip()

	# If we include all tracks, create an entry for it and populate it with all races
	if include_all_tracks:
		race_list = []
		for year_key in list(year_race_count.keys()):
			year_races = year_race_count[year_key]
			for race_id in range(1, year_races + 1, 1):
				if exclusion_list.count([year_key, race_id]) == 0:
					race_list.append({'year': year_key, 'weekend': race_id})
		track_list['All'] = {'races': race_list, 'type': 'multiple tracks'}

	# Create a variable to store track-by-track data and then loop through each track
	track_stats = {}
	for track_name in list(track_list.keys()):
		print('***** Processing ' + track_name)
		# Get the list of races and create a blank variable for the stats of each race
		race_list = track_list[track_name]['races']
		track_type = track_list[track_name]['type']
		race_stats = []

		# Loop through each race at the track
		for race_info in race_list:
			# Get the year and the weekend
			year = race_info['year']
			weekend = race_info['weekend']
			print('Year: ' + '{:d}'.format(year) + ' Weekend: ' + '{:d}'.format(weekend))

			# Load data from the race and qualifying
			race_session = fastf1.get_session(year, weekend, 'R')
			race_session.load()
			race_date = race_session.event['EventDate'].strftime('%Y-%m-%d %H:%M:%S')
			print('***** Race session is ' + race_session.event['EventName'] + ' on ' + race_session.event['EventDate'].strftime('%Y-%m-%d %H:%M:%S'))

			quali_session = fastf1.get_session(year, weekend, 'Q')
			quali_session.load()
			print('***** Qualifying session is ' + quali_session.event['EventName'] + ' on ' + quali_session.event['EventDate'].strftime('%Y-%m-%d %H:%M:%S'))

			# Get the race duration
			race_duration = (np.max(np.add(race_session.laps['Time'], race_session.laps['LapTime'])) - np.min(race_session.laps['Time'])).total_seconds()

			# Get the drivers who didn't withdraw from the race
			driver_data = race_session.results.loc[race_session.results['Status'] != 'Withdrew']

			# Get driver numbers and grid positions
			driver_keys = driver_data['DriverNumber'].copy()
			driver_start = driver_data['GridPosition'].copy()
			driver_count = driver_keys.count()

			# Find drivers that started from the pit lane
			driver_pitlane_st = driver_start[driver_start == 0.0].keys().tolist()
			driver_grid_st = driver_start[driver_start != 0.0].keys().tolist()

			# Calculate positions for drivers starting from the pit lane
			driver_grid_pos = np.add(np.argsort(driver_start[driver_grid_st]).argsort(), 1.0)
			max_grid_pos = driver_grid_pos.max()
			quali_order = quali_session.results['Position'][driver_pitlane_st].sort_values().keys().tolist()
			driver_start.loc[driver_grid_st] = driver_grid_pos
			driver_start.loc[quali_order] = np.arange(max_grid_pos + 1.0, max_grid_pos + 1.0 + len(quali_order), 1.0).tolist()
			driver_order_start = driver_start.sort_values().keys().tolist()

			# Get the total number of laps that were run
			lap_count = race_session.laps['LapNumber'].max().astype(int)

			# Create an array to store driver positions
			driver_positions = np.zeros((lap_count + 1, driver_count))
			driver_position_advances = np.zeros((lap_count + 1))
			driver_position_lap_number = np.arange(0, lap_count + 1, 1)
			driver_position_final = np.arange(1, driver_count + 1, 1)

			# Get the final order of drivers
			driver_order_finish = race_session.results.loc[driver_keys]['DriverNumber'].tolist()

			# Calculate the driver starting positions
			driver_positions_start = [driver_order_start.index(x) + 1 for x in driver_order_finish]
			driver_positions_prev = driver_positions_start.copy()
			driver_positions[0, :] = np.array(driver_positions_start)
			driver_position_advances[0] = 0

			# Analyze each lap
			for lap_num in range(1, lap_count + 1, 1):
				driver_order_lap = race_session.laps[race_session.laps['LapNumber'] == lap_num].sort_values('Position')['DriverNumber'].tolist()
				driver_positions_lap = [driver_order_lap.index(x) + 1 for x in driver_order_finish if driver_order_lap.count(x) == 1]
				driver_positions_lap += list(range(len(driver_positions_lap) + 1, driver_count + 1, 1))
				driver_positions[lap_num, :] = np.array(driver_positions_lap)
				driver_position_advances[lap_num] = np.sum(np.clip(np.subtract(np.array(driver_positions_prev), np.array(driver_positions_lap)), 0, None))
				driver_positions_prev = driver_positions_lap.copy()

			# Store the data and append it to the list of stats for all races at this track
			race_data = {'year': year, 'weekend': weekend, 'lap_count': lap_count, 'driver_count': driver_count, 'driver_position_lap_number': driver_position_lap_number, 'driver_position_advances': driver_position_advances, 'driver_position_final': driver_position_final, 'driver_positions': driver_positions, 'rel_driver_position_lap_number': np.divide(driver_position_lap_number, lap_count), 'rel_driver_position_advances': np.divide(driver_position_advances, driver_count), 'rel_driver_position_final': np.divide(np.subtract(driver_position_final, 1), driver_count - 1), 'rel_driver_positions': np.divide(np.subtract(driver_positions, 1), driver_count - 1), 'date': race_date, 'duration': race_duration}
			race_stats.append(race_data)

		# Create empty arrays for the data
		race_times = np.arange(0, 1 + calc_frequency, calc_frequency)
		race_pos_laps = np.zeros(race_times.shape)
		race_pos_laps_mean = np.zeros(race_times.shape)
		race_pos_corr = np.zeros(race_times.shape)
		race_pos_pval = np.zeros(race_times.shape)
		race_pos_leverage = np.zeros(race_times.shape)
		race_pos_advancement = np.zeros(race_times.shape)
		race_pos_excitement = np.zeros(race_times.shape)

		# Loop through each time of the race
		for race_time_idx in range(0, race_times.shape[0], 1):
			# Get the (normalized) start and end time of the interval
			race_time_center = race_times[race_time_idx]
			race_time_begin = race_time_center - (calc_interval / 2)
			race_time_end = race_time_center + (calc_interval / 2)

			# Set up some blank lists for data as we traverse the interval
			race_pos_cur_list = []
			race_pos_final_list = []
			race_pos_advances_list = []

			# Loop through each race and populate the lists we just declared
			for race_data in race_stats:
				lap_idx_list = np.logical_and(np.greater_equal(race_data['rel_driver_position_lap_number'], race_time_begin), np.less_equal(race_data['rel_driver_position_lap_number'], race_time_end)).nonzero()
				race_pos_advances_list.extend(np.squeeze(np.multiply(race_data['rel_driver_position_advances'][lap_idx_list], (3600 / race_data['duration']) * race_data['lap_count'] / (1 / calc_frequency))).tolist())
				race_pos_arr = race_data['rel_driver_positions'][np.squeeze(lap_idx_list), :]
				for pos_idx in range(0, race_pos_arr.shape[0], 1):
					race_pos_cur_list.extend(np.squeeze(race_pos_arr[pos_idx, :]).tolist())
					race_pos_final_list.extend(np.squeeze(race_data['rel_driver_position_final']).tolist())

			# Now calculate some statistics on the data we just stored in the lists
			race_pos_laps[race_time_idx] = len(race_pos_advances_list)
			race_pos_laps_mean[race_time_idx] = len(race_pos_advances_list) / len(race_stats)
			race_pos_advancement[race_time_idx] = np.mean(np.array(race_pos_advances_list))
			reg = stats.linregress(np.array(race_pos_cur_list), np.array(race_pos_final_list))
			race_pos_corr[race_time_idx] = reg.rvalue
			race_pos_pval[race_time_idx] = reg.pvalue
			race_pos_leverage[race_time_idx] = abs(reg.rvalue)
			race_pos_excitement[race_time_idx] = abs(reg.rvalue) * np.mean(np.array(race_pos_advances_list))

		# Store the overall data for this track
		cur_track_data = {
			'races': len(race_stats),
			'laps_used': race_pos_laps,
			'laps_per_race_used': race_pos_laps_mean,
			'advancement': race_pos_advancement,
			'correlation': race_pos_corr,
			'pvalue': race_pos_pval,
			'leverage': race_pos_leverage,
			'excitement': race_pos_excitement,
			'race_stats': race_stats,
			'track_type': track_type
		}
		track_stats[track_name] = cur_track_data

	# Calculate the percentage of time through the race for each interval
	race_times_pct = np.multiply(race_times, 100)

	export_data = {
		'track_stats': track_stats,
		'race_times': race_times,
		'race_times_pct': race_times_pct,
		'calc_frequency': calc_frequency,
		'calc_interval': calc_interval,
		'series': 'formula1'
	}

	pickle_handle = open(output_file_name, 'wb')
	pickle.dump(export_data, pickle_handle)
	pickle_handle.close()

if __name__ == '__main__':
	main()

