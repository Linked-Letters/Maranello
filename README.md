# Maranello
This is a collection of tools for basic analysis of publicly-available data sets from professional racing series.

# Tools

**f1_team_pace.py** *\<season\> \<output data file\> [number of races to process at the start of the season]*\
This processes Formula 1 data from one season and stores it in an output file for postprocessing.

**pace_postprocess.py** *\<input data file\> \<output image name\>*\
This generates an image showing the pace of teams over the course of a season, using the output from f1_team_pace.py.

**f1_leverage.py** *\<output data file\>*\
This downloads and processes data from the races specified in the source file, then outputs the results to be plotted later.

**nascar_leverage.py** *\<first season\> \<last season\> \<series 1=Cup, 2=Xfinity, 3=Truck\> \<output data file\>*\
This downloads and processes data from the requested years in the series, then outputs the results to be plotted later.

**leverage_postprocess.py** *\<input data file\> [scale factor for advancement and excitement, defaults to 1.0]*\
This plots the data in the input file, which is generated either by f1_leverage.py or nascar_leverage.py.

# Requirements
This is a collection pf Python tools, and they use the following libraries:
* fastf1
* matplotlib
* numpy
* pandas
* scikit-learn
* scipy

# License
Copyright (C) 2025 George Limpert

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see [https://www.gnu.org/licenses/](https://www.gnu.org/licenses/).
