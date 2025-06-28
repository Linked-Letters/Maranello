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

import datetime
import fastf1
import math
import numpy as np
import pandas as pd
import pickle
import sklearn
import sys

# This is the maximum time gap in seconds to the next car for a lap to be considered obstructed
obstructed_timediff = 1.5

# This is the minimum number of laps to use in pace analysis
min_analysis_laps = 150

# This is the minimum number of laps for each team on the tyre
min_team_laps = 15

# This is the minimum proportion of the field that used this tyre at least the minimum laps, otherwise we won't analyze the data for this tyre
min_field_proportion = 0.6

def main ():
	global obstructed_timediff, min_analysis_laps, min_team_laps, min_field_proportion

	# Get the parameters from the command line
	if len(sys.argv) < 3:
		print('Usage: '+sys.argv[0]+' <season> <output file> [number of races]')
		exit()
	try:
		season = int(sys.argv[1].strip())
	except:
		print('Invalid season')
		exit()
	output_file_name = sys.argv[2].strip()
	try:
		schedule_data = fastf1.get_event_schedule(season)
		season_races = max(schedule_data['RoundNumber'].to_list())
		if len(sys.argv) >= 4:
			req_races = int(sys.argv[3].strip())
			if req_races > season_races:
				races_to_use = season_races
			elif req_races <= 0:
				print('Need at least one race for analysis')
				exit()
			else:
				races_to_use = req_races
		else:
			races_to_use = season_races
	except:
		print('Error in number of races')
		exit()

	season_performance_data = {'season': season, 'races': {}}

	# Loop through each race
	for weekend in range(1, races_to_use + 1, 1):
		# Load the race session
		race_session = fastf1.get_session(season, weekend, 'R')
		race_session.load()

		# Get the tyre compounds used during the race
		tyre_compounds_used = list(set(list(race_session.laps['Compound'])))
		session_data = race_session.results.loc[race_session.results['Status'] != 'Withdrew']
		teams_in_session = list(set(list(session_data['TeamName'])))

		race_tyre_performance = {}

		# Estimate the typical pace around the track by averaging all the laps that aren't affected by yellow flags, pitting, are deleted, or are otherwise suspect
		track_lap_speed = np.mean(np.array([x.total_seconds() for x in race_session.laps[(race_session.laps['TrackStatus'] == '1') & pd.isnull(race_session.laps['PitOutTime']) & pd.isnull(race_session.laps['PitInTime']) & (~race_session.laps['Deleted']) & (~race_session.laps['FastF1Generated']) & (race_session.laps['IsAccurate'])]['LapTime'].to_list()]))

		# Loop through each tyre compound
		for tyre_compound in tyre_compounds_used:
			# For the tyre compound, remove pit in/out laps, laps that aren't green flag laps, deleted laps, and anything else that's not accurate
			fast_laps = race_session.laps[(race_session.laps['Compound'] == tyre_compound) & (race_session.laps['TrackStatus'] == '1') & pd.isnull(race_session.laps['PitOutTime']) & pd.isnull(race_session.laps['PitInTime']) & (~race_session.laps['Deleted']) & (~race_session.laps['FastF1Generated']) & (race_session.laps['IsAccurate']) & (~pd.isnull(race_session.laps['TyreLife'])) & (~pd.isnull(race_session.laps['Compound']))]

			# Try to filter out laps where a car had another in front of it that it was racing for position
			was_unobstructed = []
			for index, lap in fast_laps.iterrows():
				lap_lapnumber = lap['LapNumber']
				lap_lapdriver = lap['Driver']
				# Calculate the start time differential compared to all laps, then keep laps 
				lap_timediff = lap['LapStartTime'] - race_session.laps[(race_session.laps['Driver'] != lap_lapdriver) | (race_session.laps['LapNumber'] != lap_lapnumber)]['LapStartTime']
				lap_timediff = lap_timediff[lap_timediff > datetime.timedelta(0)]
				if len(lap_timediff) == 0:
					was_unobstructed.append(True)
				elif min(lap_timediff).total_seconds() <= obstructed_timediff:
					was_unobstructed.append(False)
				else:
					was_unobstructed.append(True)
			unobstructed_fast_laps = fast_laps[was_unobstructed]

			# If there are enough laps left after the filtering, then analyze them
			if len(unobstructed_fast_laps) >= min_analysis_laps:
				# Calculate how many laps each team ran on a tyre, then verify that enough teams ran the tyre for enough laps that we can hopefully do meaningful analysis
				team_lap = [x for x in list(unobstructed_fast_laps['Team'])]
				laps_per_team = {}
				for team_name in teams_in_session:
					laps_per_team[team_name] = team_lap.count(team_name)
				team_use_proportion = sum([int(x >= min_team_laps) for x in list(laps_per_team.values())]) / len(teams_in_session)
				# If enough teams have used the tyre, then analyze it
				if team_use_proportion >= min_field_proportion:
					# Build a regression model based on the lap number (proxy for fuel load) and the tire usage
					tyre_lap = [x for x in list(unobstructed_fast_laps['TyreLife'])]
					lap_times = [x.total_seconds() for x in list(unobstructed_fast_laps['LapTime'])]
					lap_numbers = [x for x in list(unobstructed_fast_laps['LapNumber'])]
					indep_vars = np.transpose(np.array([np.square(np.array(tyre_lap)).tolist(), tyre_lap, lap_numbers]))
					dep_var = np.array(lap_times)
					laptime_model = sklearn.linear_model.LinearRegression()
					laptime_model.fit(indep_vars, dep_var)
					# For each team that used the tyre, calculate their difference from the predicted lap time
					team_performance_data = {}
					for team_name in teams_in_session:
						if laps_per_team[team_name] == 0:
							team_performance_data[team_name] = {'laps': 0, 'predicted': np.array([]), 'actual': np.array([]), 'mean_differential': np.nan}
						else:
							predicted_times = []
							actual_times = []
							laps_used = 0
							for index, lap in unobstructed_fast_laps[unobstructed_fast_laps['Team'] == team_name].iterrows():
								laps_used += 1
								actual_times.append(lap['LapTime'].total_seconds())
								predicted_times.append(laptime_model.predict([[np.square(np.array(lap['TyreLife'])).tolist(), lap['TyreLife'], lap['LapNumber']]])[0])
							team_performance_data[team_name] = {'laps': laps_used, 'predicted': np.array(predicted_times), 'actual': np.array(actual_times), 'mean_differential': np.mean(np.subtract(np.array(actual_times), np.array(predicted_times)))}
					race_tyre_performance[tyre_compound] = team_performance_data

		# Now, merge together the performance data for each team with all tyre compounds (provided there's enough data)
		team_performance = {}
		for team_name in teams_in_session:
			actual_times = []
			predicted_times = []
			for tyre_compound in list(race_tyre_performance.keys()):
				actual_times.extend(race_tyre_performance[tyre_compound][team_name]['actual'].tolist())
				predicted_times.extend(race_tyre_performance[tyre_compound][team_name]['predicted'].tolist())
			if len(actual_times) == 0:
				team_performance[team_name] = {'time': math.nan, 'percent': math.nan, 'laps': 0, 'time_stdev': math.nan, 'percent_stdev': math.nan, 'actual': actual_times, 'predicted': predicted_times}
			else:
				lap_differential_data = np.subtract(np.array(actual_times), np.array(predicted_times))
				lap_differential = np.mean(lap_differential_data)
				lap_differential_stdev = np.std(lap_differential_data)
				lap_percent = lap_differential * 100.0 / track_lap_speed
				lap_percent_stdev = lap_differential_stdev * 100.0 / track_lap_speed
				team_performance[team_name] = {'time': lap_differential, 'percent': lap_percent, 'laps': len(actual_times), 'time_stdev': lap_differential_stdev, 'percent_stdev': lap_percent_stdev, 'actual': actual_times, 'predicted': predicted_times}

		# Store the data in a data structure
		season_performance_data['races'][weekend] = {'reference_lap': track_lap_speed, 'tyre_data': race_tyre_performance, 'team_data': team_performance, 'round': race_session.event['RoundNumber'], 'country': race_session.event['Country'], 'location': race_session.event['Location'], 'race_name': race_session.event['EventName'], 'race_date': race_session.event['EventDate']}

	# Output the data for later use
	pickle_handle = open(output_file_name, 'wb')
	pickle.dump(season_performance_data, pickle_handle)
	pickle_handle.close()

if __name__ == '__main__':
	main()
