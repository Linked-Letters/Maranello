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

import json
import numpy as np
import pickle
import requests
import scipy.stats as stats
import sys
import time

# Frequency (how often through the race do we determine the stats) and interval of race calculations (the window over which the data are averaged when doing the calculations)
calc_frequency = 0.01
calc_interval = 0.1

track_lookup_table = {
	'u4': {'type': 'intermediate', 'name': 'Darlington'},
	'u14': {'type': 'short track', 'name': 'Bristol'},
	'u22': {'type': 'short track', 'name': 'Martinsville'},
	'u26': {'type': 'short track', 'name': 'Richmond'},
	'u34': {'type': 'road course', 'name': 'Road America'},
	'u38': {'type': 'intermediate', 'name': 'Fontana'},
	'u39': {'type': 'intermediate', 'name': 'Chicagoland'},
	'u40': {'type': 'intermediate', 'name': 'Homestead'},
	'u41': {'type': 'intermediate', 'name': 'Kansas'},
	'u42': {'type': 'intermediate', 'name': 'Las Vegas'},
	'u43': {'type': 'intermediate', 'name': 'Texas'},
	'u45': {'type': 'intermediate', 'name': 'Gateway'},
	'u47': {'type': 'short track', 'name': 'Indianapolis Raceway Park'},
	'u51': {'type': 'intermediate', 'name': 'Milwaukee Mile'},
	'u52': {'type': 'intermediate', 'name': 'Nashville'},
	'u61': {'type': 'intermediate', 'name': 'Kentucky'},
	'u72': {'type': 'road course', 'name': 'Mid-Ohio'},
	'u75': {'type': 'road course', 'name': 'Mexico City'},
	'r82': {'type': 'superspeedway', 'name': 'Talladega'},
	'u84': {'type': 'intermediate', 'name': 'Phoenix'},
	'u99': {'type': 'road course', 'name': 'Sonoma'},
	'u103': {'type': 'intermediate', 'name': 'Dover'},
	'r105': {'type': 'superspeedway', 'name': 'Daytona'},
	'r111': {'type': 'superspeedway', 'name': 'Atlanta'},
	'u111': {'type': 'intermediate', 'name': 'Atlanta (old)'},
	'u123': {'type': 'intermediate', 'name': 'Indianapolis'},
	'u133': {'type': 'intermediate', 'name': 'Michigan'},
	'u138': {'type': 'intermediate', 'name': 'New Hampshire'},
	'u157': {'type': 'road course', 'name': 'Watkins Glen'},
	'u159': {'type': 'short track', 'name': 'Bowman Gray'},
	'u162': {'type': 'intermediate', 'name': 'Charlotte'},
	'u175': {'type': 'intermediate', 'Name': 'Rockingham'},
	'u177': {'type': 'short track', 'name': 'North Wilkesboro'},
	'u198': {'type': 'intermediate', 'name': 'Pocono'},
	'u204': {'type': 'road course', 'name': 'Portland'},
	'u206': {'type': 'short track', 'name': 'Iowa'},
	'u208': {'type': 'dirt track', 'name': 'Eldora'},
	'u209': {'type': 'road course', 'name': 'Bowmanville'},
	'u210': {'type': 'road course', 'name': 'Charlotte Roval'},
	'u211': {'type': 'road course', 'name': 'Indianapolis Road Course'},
	'u212': {'type': 'road course', 'name': 'Daytona Road Course'},
	'u214': {'type': 'road course', 'name': 'Circuit of the Americas'},
	'u215': {'type': 'dirt track', 'name': 'Knoxville'},
	'u216': {'type': 'dirt track', 'name': 'Bristol (dirt)'},
	'u217': {'type': 'short track', 'name': 'Los Angeles Memorial Coliseum'},
	'u218': {'type': 'street course', 'name': 'Chicago Street Course'},
	'u220': {'type': 'road course', 'name': 'Lime Rock Park'}
}

def get_track_info (track_id, restrictor_plate):
	global track_lookup_table
	if restrictor_plate:
		lookup_name = 'r' + '{:d}'.format(track_id)
	else:
		lookup_name = 'u' + '{:d}'.format(track_id)
	if list(track_lookup_table.keys()).count(lookup_name) > 0:
		return track_lookup_table[lookup_name]
	else:
		return {'type': 'unknown', 'name': 'Unknown'}

def retrieve_response_from_url (input_page_url, request_delay_time = 4, max_requests = 10):
	# Attempt to download the page
	downloaded = False
	end_requests = False
	request_count = 0
	while end_requests == False:
		server_response = requests.get(input_page_url)
		request_count = request_count + 1
		if 200 <= server_response.status_code <= 299:
			end_requests = True
			downloaded = True
		elif request_count >= max_requests:
			end_requests = True
		else:
			time.sleep(request_delay_time)

	# Either return the text or None
	if downloaded:
		return server_response
	else:
		return None

# Parse the arguments on the command line
def parse_command_line_args ():
	if len(sys.argv) < 5:
		print('Usage: ' + sys.argv[0] + ' <start year> <end year> <series> <output file>')
		sys.exit(1)
	try:
		start_year = int(sys.argv[1].strip())
		end_year = int(sys.argv[2].strip())
	except:
		print('Invalid start or end year')
		sys.exit(2)
	try:
		series_id = int(sys.argv[3].strip())
	except:
		print('Invalid series')
		sys.exit(2)
	output_file_name = sys.argv[4].strip()
	return start_year, end_year, series_id, output_file_name

def main ():
	global calc_frequency, calc_interval
	start_year, end_year, series_id, output_file_name = parse_command_line_args()
	# Create an empty data structure for track statistics
	track_stats = {}
	# Loop through the series and get a list races each year
	for cur_year in range(start_year, end_year + 1, 1):
		# Retrieve the list of races
		race_list_url = 'https://cf.nascar.com/cacher/' + '{:d}'.format(cur_year) + '/race_list_basic.json'
		race_list_rsp = retrieve_response_from_url(race_list_url)
		# If we have data, try to parse it
		if race_list_rsp is not None:
			try:
				race_list_data = json.loads(race_list_rsp.text)
			except:
				print('Error parsing race list for year ' + '{:d}'.format(cur_year))
				race_list_data = None
			# If we have a list of races, let's iterate through the races, ignoring non-points races
			if race_list_data is not None:
				series_race_list = race_list_data['series_' + '{:d}'.format(series_id)]
				r_race_weekend = 0
				for race_desc in series_race_list:
					# Only include points races
					if race_desc['race_type_id'] == 1:
						# Get some basic data about the race
						r_race_weekend += 1
						r_race_id = race_desc['race_id']
						r_series_id = race_desc['series_id']
						r_race_season = race_desc['race_season']
						r_track_id = race_desc['track_id']
						r_restrictor_plate = race_desc['restrictor_plate']
						r_track_name = race_desc['track_name']
						r_race_date = race_desc['race_date']
						r_race_name = race_desc['race_name']
						r_actual_laps = race_desc['actual_laps']
						r_scheduled_laps = race_desc['scheduled_laps']
						track_info = get_track_info(r_track_id, r_restrictor_plate)
						r_track_short_name = track_info['name']
						r_track_type = track_info['type']
						r_total_race_time = race_desc['total_race_time']
						r_race_duration = 0
						race_time_split = r_total_race_time.split(':')
						for t_pos in range(1, len(race_time_split) + 1, 1):
							r_race_duration += ((60 ** (t_pos - 1)) * int(race_time_split[-t_pos]))
						# Retrieve the lap data for the race
						race_lap_data_url = 'https://cf.nascar.com/cacher/' + '{:d}'.format(r_race_season) + '/' + '{:d}'.format(r_series_id) + '/' + '{:d}'.format(r_race_id) + '/lap-times.json'
						print('***** Race session is ' + r_race_name + ' at ' + r_track_name + ' on ' + r_race_date)
						race_lap_data_rsp = retrieve_response_from_url(race_lap_data_url)
						# As with before, if we have data, try to parse it
						if race_lap_data_rsp is not None:
							try:
								race_lap_data = json.loads(race_lap_data_rsp.text)
							except:
								print('Error parsing race ' + '{:d}'.format(r_race_id) + ' in year ' + '{:d}'.format(r_race_season))
								race_lap_data = None
						# If we have data, let's analyze it
						if race_lap_data is not None:
							driver_count = len(race_lap_data['laps'])
							driver_position_final = np.array([x['RunningPos'] for x in race_lap_data['laps']])
							cars_running_order = []
							# Loop through the lap data for each car and try to build a position history
							for car_lap_data in race_lap_data['laps']:
								# Begin with the running order on the pace laps
								prev_lap = car_lap_data['Laps'][0]['Lap']
								car_running_position = [car_lap_data['Laps'][0]['RunningPos']] * (prev_lap + 1)
								# Then loop through the data for the additional laps, accounting for that laps are occasionally omitted
								for current_lap_data in car_lap_data['Laps'][1:]:
									current_lap = current_lap_data['Lap']
									car_running_position += ([current_lap_data['RunningPos']] * (current_lap - prev_lap))
									prev_lap = current_lap
								# If the car retires from the race, it won't have complete data, so put its finishing position in for all remaining laps
								if len(car_running_position) < (r_actual_laps + 1):
									car_running_position += ([car_lap_data['RunningPos']] * ((r_actual_laps + 1) - len(car_running_position)))
								# In some instances, it seems there can be an extra lap added incorrectly at the end of a race, and we'll need to detect and remove this
								if len(car_running_position) > (r_actual_laps + 1):
									car_running_position = car_running_position[0: r_actual_laps + 1]
								# Append it to the list of positions per car, which we'll need to transpose later
								cars_running_order.append(car_running_position)
							# This transposes the array so that the first axis is the lap instead of the car, which makes it easier to work with
							driver_positions = np.transpose(np.array(cars_running_order))
							# Calculate the number of positions advanced per lap for the entire field
							driver_position_advances = np.zeros((r_actual_laps + 1))
							for current_lap in range(1, r_actual_laps + 1, 1):
								driver_position_advances[current_lap] = np.sum(np.clip(np.subtract(driver_positions[current_lap - 1, :], driver_positions[current_lap, :]), 0, None))
							# Calculate additional stats about lap numbers and some other data
							driver_position_lap_number = np.arange(0, r_actual_laps + 1, 1)
							rel_driver_position_lap_number = np.divide(driver_position_lap_number, r_actual_laps)
							rel_driver_position_advances = np.divide(driver_position_advances, driver_count)
							rel_driver_position_final = np.divide(np.subtract(driver_position_final, 1), driver_count - 1)
							rel_driver_positions = np.divide(np.subtract(driver_positions, 1), driver_count - 1)
							race_data = {'year': r_race_season, 'weekend': r_race_weekend, 'lap_count': r_actual_laps, 'scheduled_laps': r_scheduled_laps, 'driver_count': driver_count, 'driver_position_lap_number': driver_position_lap_number, 'driver_position_advances': driver_position_advances, 'driver_position_final': driver_position_final, 'driver_positions': driver_positions, 'rel_driver_position_lap_number': rel_driver_position_lap_number, 'rel_driver_position_advances': rel_driver_position_advances, 'rel_driver_position_final': rel_driver_position_final, 'rel_driver_positions': rel_driver_positions, 'date': r_race_date, 'race_name': r_race_name, 'track_id': r_track_id, 'restrictor_plate': r_restrictor_plate, 'series_id': r_series_id, 'duration': r_race_duration}
							# See if we need to add the track to the data structure for storing data, and if so, create it
							if list(track_stats.keys()).count(r_track_short_name) == 0:
								track_stats[r_track_short_name] = {}
								track_stats[r_track_short_name]['track_type'] = r_track_type
								track_stats[r_track_short_name]['race_stats'] = []
							# Then put the race stats in the data structure
							track_stats[r_track_short_name]['race_stats'].append(race_data)
							
						# Wait a second before retrieving the next file
						time.sleep(2)
					
		# No data, so report an error
		else:
			print('Cannot retrieve races for year ' + '{:d}'.format(cur_year))
		# Wait a second before retrieving the next file
		time.sleep(2)

	# Now, we need to go track by track and calculate the statistics
	race_times = np.arange(0, 1 + calc_frequency, calc_frequency)
	for track_idx in list(track_stats.keys()):
		# Set up some empty arrays for the data
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
			for race_data in track_stats[track_idx]['race_stats']:
				lap_idx_list = np.logical_and(np.greater_equal(race_data['rel_driver_position_lap_number'], race_time_begin), np.less_equal(race_data['rel_driver_position_lap_number'], race_time_end)).nonzero()
				race_pos_advances_list.extend(np.squeeze(np.multiply(race_data['rel_driver_position_advances'][lap_idx_list], (3600 / race_data['duration']) * race_data['lap_count'] / (1 / calc_frequency))).tolist())
				race_pos_arr = race_data['rel_driver_positions'][np.squeeze(lap_idx_list), :]
				for pos_idx in range(0, race_pos_arr.shape[0], 1):
					race_pos_cur_list.extend(np.squeeze(race_pos_arr[pos_idx, :]).tolist())
					race_pos_final_list.extend(np.squeeze(race_data['rel_driver_position_final']).tolist())

			# Now calculate some statistics on the data we just stored in the lists
			race_count = len(track_stats[track_idx]['race_stats'])
			race_pos_laps[race_time_idx] = len(race_pos_advances_list)
			race_pos_laps_mean[race_time_idx] = len(race_pos_advances_list) / race_count
			race_pos_advancement[race_time_idx] = np.mean(np.array(race_pos_advances_list))
			reg = stats.linregress(np.array(race_pos_cur_list), np.array(race_pos_final_list))
			race_pos_corr[race_time_idx] = reg.rvalue
			race_pos_pval[race_time_idx] = reg.pvalue
			race_pos_leverage[race_time_idx] = abs(reg.rvalue)
			race_pos_excitement[race_time_idx] = abs(reg.rvalue) * np.mean(np.array(race_pos_advances_list))

		# Store the data we just calculated
		track_stats[track_idx]['races'] = race_count
		track_stats[track_idx]['laps_used'] = race_pos_laps
		track_stats[track_idx]['laps_per_race_used'] = race_pos_laps_mean
		track_stats[track_idx]['advancement'] = race_pos_advancement
		track_stats[track_idx]['correlation'] = race_pos_corr
		track_stats[track_idx]['pvalue'] = race_pos_pval
		track_stats[track_idx]['leverage'] = race_pos_leverage
		track_stats[track_idx]['excitement'] = race_pos_excitement

	# Calculate the percentage of time through the race for each interval
	race_times_pct = np.multiply(race_times, 100)

	# Now we need to create a final output structure with all of the relevant data
	export_data = {
		'track_stats': track_stats,
		'race_times': race_times,
		'race_times_pct': race_times_pct,
		'calc_frequency': calc_frequency,
		'calc_interval': calc_interval,
		'series': 'nascar' + '{:d}'.format(series_id)
	}

	pickle_handle = open(output_file_name, 'wb')
	pickle.dump(export_data, pickle_handle)
	pickle_handle.close()

if __name__ == '__main__':
	main()
