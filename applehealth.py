import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import time
from datetime import datetime
import pytz

# an instance of apple Health
# fname is the name of data file to be parsed must be an XML files
# flags for cache
class AppleHealth:
    def __init__(self, fname = 'export.xml', pivotIndex = 'endDate', readCache = False, writeCache = False):
        #check cache flag and cache accordingly
        if readCache:
            self.readCache()
            if writeCache:
                self.cacheAll()
            return

        # create element tree object
        s = time.time()
        tree = ET.parse(fname)
        e = time.time()
        print("Tree parsing Time = {}".format(e-s))


        # for every health record, extract the attributes into a dictionary (columns). Then create a list (rows).
        s = time.time()
        root = tree.getroot()
        record_list = [x.attrib for x in root.iter('Record')]
        workout_list = [x.attrib for x in root.iter('Workout')]
        e = time.time()
        print("record list Time = {}".format(e-s))

        # create DataFrame from a list (rows) of dictionaries (columns)
        s = time.time()
        self.record_data = pd.DataFrame(record_list)
        self.workout_data = pd.DataFrame(workout_list)
        print(self.workout_data)
        e = time.time()
        print("creating DF Time = {}".format(e-s))

        format = '%Y-%m-%d %H:%M:%S'
        # proper type to dates
        def get_split_date(strdt):
            split_date = strdt.split()
            str_date = split_date[1] + ' ' + split_date[2] + ' ' + split_date[5] + ' ' + split_date[3]
            return str_date
        s = time.time()
        for col in ['creationDate', 'startDate', 'endDate']:
            self.record_data[col] = pd.to_datetime(self.record_data[col], format=format)
            self.workout_data[col] = pd.to_datetime(self.workout_data[col], format=format)
        e = time.time()
        print("date conv Time = {}".format(e-s))

        s = time.time()
        # value is numeric, NaN if fails
        self.record_data['value'] = pd.to_numeric(self.record_data['value'], errors='coerce')
        self.workout_data['duration'] = pd.to_numeric(self.workout_data['duration'], errors='coerce')
        self.workout_data['totalDistance'] = pd.to_numeric(self.workout_data['totalDistance'], errors='coerce')
        self.workout_data['totalEnergyBurned'] = pd.to_numeric(self.workout_data['totalEnergyBurned'], errors='coerce')

        # some records do not measure anything, just count occurences
        # filling with 1.0 (= one time) makes it easier to aggregate
        self.record_data['value'] = self.record_data['value'].fillna(1.0)

        # shorter observation names: use vectorized replace function
        self.record_data['type'] = self.record_data['type'].str.replace('HKQuantityTypeIdentifier', '')
        self.record_data['type'] = self.record_data['type'].str.replace('HKCategoryTypeIdentifier', '')
        self.workout_data['workoutActivityType'] = self.workout_data['workoutActivityType'].str.replace('HKWorkoutActivityType', '')
        print(self.workout_data)
        e = time.time()
        print("rest Time = {}".format(e-s))

        # pivot
        s = time.time()
        self.pivot_record_df = self.record_data.pivot_table(index='endDate', columns='type', values='value')
        self.pivot_workout_df = self.workout_data.pivot_table(index='endDate', columns='workoutActivityType', values=['duration', 'totalDistance', 'totalEnergyBurned'])
        # self.pivot_record_df.tz_convert('UTC', level=0)
        # self.pivot_workout_df.tz_convert('UTC', level=0)
        print(self.pivot_workout_df)
        # print(self.pivot_workout_df.xs('Soccer'))
        print()
        e = time.time()
        print("pivot Time = {}".format(e-s))

        if writeCache:
            self.cacheAll()

    # resample the record dataframe to period and perform calculations to metrics listed in dict
    def resampleRecords(self, resampDict, period = 'D'):
        self.record_df = self.pivot_record_df.resample(period).agg(resampDict)

    # resample the workout dataframe to period and perform calculations to metrics listed in dict
    def resampleWorkouts(self, resampDict, period = 'D'):
        self.workout_df = self.pivot_workout_df.resample(period).agg(resampDict)

    # calculate the delta of deltaCol
    # TODO: Test
    def delta(self, deltaCol):
        self.record_df['Delta'] = df[deltaCol].diff()

    # display a time time series of the provided timeSeriesCol
    # TODO: allow for selecting of range of dates
    def timeSeries(self, timeSeriesCol, col = 'purple', lw = 1, fsx = 12, fsy = 4):
        fig = plt.figure(figsize=(fsx,fsy))
        sns.lineplot(data=self.record_df[timeSeriesCol], color=col, linewidth=lw)
        plt.show()

    # display workout trends of selected workout ployyed with metrics
    def workoutTrends(self, workout_type, metrics = ['duration', 'totalDistance', 'totalEnergyBurned'], col = 'purple', lw = 1, fsx = 12, fsy = 4):
        fig = plt.figure(figsize=(fsx,fsy))
        sns.lineplot(data=self.pivot_workout_df.loc[:,(metrics, workout_type)], color=col, linewidth=lw)
        plt.show()

    # returns a correlation matrix of the record dataframe
    def correlationMatrix(self):
        # correlation matrix
        self.cm = self.record_df.corr()
        return self.cm

    # displays the correlation matrix cm as a heat map
    def heatMap(self, fsx = 8, fsy = 6):
        # correlation matrix
        self.cm = self.record_df.corr()
        # heatmap
        fig = plt.figure(figsize=(fsx, fsy))
        sns.heatmap(self.cm, annot=True, fmt=".2f", vmin=-1.0, vmax=+1.0, cmap='Spectral')
        plt.show()

    # displays a pair plot of the record dataframe
    def pairPlot(self, pairPlotArgs, fsx = 12, fsy = 12, col = 'purple', lw = 1,):
        fig = plt.figure(figsize=(fsx, fsy))

        g = sns.pairplot(self.record_df[pairPlotArgs],
                     kind='kde',
                     plot_kws=dict(fill=False, color=col, linewidths=lw),
                     diag_kws=dict(fill=False, color=col, linewidth=lw))

        # add observation dots
        g.map_offdiag(sns.scatterplot, marker='.', color='black')
        plt.show()

    # returns a slice of the pivot record data frame
    def narrowRange(self, startDate, endDate):
        return self.pivot_record_df[startDate:endDate]

    # returns the resampled chosen metric of the pivot record dataframe to a daily sum
    def dayOnDay(self, metric):
        self.by_day = self.pivot_record_df[metric].resample('D').sum()
        return self.by_day

    # returns a monthly average of daily sum of chosen metric
    def means_by_month(self, metric):
        means_by_distinct_month = self.dayOnDay(metric).resample('M').mean()
        means_by_month = means_by_distinct_month.groupby(means_by_distinct_month.index.month).mean()
        means_by_month.index = list(calendar.month_name)[1:]
        return means_by_month

    # displays a monthly bar graph of daily average of metric
    def monthlyBar(self, metric):
        self.means_by_month(metric).plot(kind='bar')
        plt.show()

    # writes all the records of workout_type and selected metrics to a csv file fname
    def extractWorkoutType(self, workout_type, fname, metrics = ['duration', 'totalDistance', 'totalEnergyBurned']):
        self.pivot_workout_df.loc[:,(metrics, workout_type)].to_csv(fname)

    # writes all all workouts to a csv file fname
    def extractAllWorkout(self, fname):
        self.pivot_workout_df.to_csv(fname)

    # writes all all records to a csv file fname
    def extractAllRecords(self, fname):
        self.pivot_record_df.to_csv(fname)

    # writes all specific records of record_type to a csv file fname
    def extractRecordType(self, record_type, fname):
        self.pivot_record_df[record_type].to_csv(fname)

    # generates sleep dataframe to store sleep analysis data
    def sleepRecord(self):
        self.sleep_df = self.pivot_record_df['SleepAnalysis']

    # displays a time series of daily sleep hours
    def sleepAnalysis(self, col = 'purple', lw = 1, fsx = 12, fsy = 4):
        fig = plt.figure(figsize=(fsx,fsy))
        sns.lineplot(data=self.sleep_df.resample('D').sum(), color=col, linewidth=lw)
        plt.show()

    # writes pivot record and workout dataframe to pkl files of associated names
    # TODO: Error handling
    def cacheAll(self):
        self.pivot_record_df.to_pickle('cached_pivot_record_df.pkl')
        self.pivot_workout_df.to_pickle('cached_pivot_workout_df.pkl')

    # reads cache to populate record dataframe and workout dataframe
    # TODO: Error handling
    def readCache(self):
        self.pivot_record_df = pd.read_pickle('cached_pivot_record_df.pkl')
        self.pivot_workout_df = pd.read_pickle('cached_pivot_workout_df.pkl')

    # investigate use case
    # TODO: Test
    def dropNullRecord(self, subset = None, thresh = 0.0, axis = 0, inplace=False):
        print(int( 0.000001 * self.pivot_record_df.shape[0]))
        return self.pivot_record_df.dropna(thresh  = int(thresh * len(self.pivot_record_df.index)), axis = axis , inplace = inplace, subset = subset)

    # compress records over period into 1 record by calculating mean of each record collected
    # TODO: Test
    def compressRecord(self, period = '5T'):
        return self.pivot_record_df.resample(period).mean()

    # find the delta between each record collected
    def avgRecordFreq(self):
        return self.pivot_record_df.index.to_series().diff().mean()

    # find standard deviation of all records
    def standardDeviationRecord(self):
        return self.pivot_record_df.std()

    # find standard deviation of all metrics between workouts
    def standardDeviationWorkout(self):
        return self.pivot_workout_df.std()

    # find average energy burned for specific workout_type
    # TODO: Rename
    def workoutAvgEnergyBurnedPerMin(self, workout_type):
        return self.pivot_workout_df['totalEnergyBurned'][workout_type].sum() / self.pivot_workout_df['duration'][workout_type].sum()

    # find average distance traveled for specific workout_type
    # TODO: Rename
    def workoutAvgDistancePerMin(self, workout_type):
        return self.pivot_workout_df['totalDistance'][workout_type].sum() / self.pivot_workout_df['duration'][workout_type].sum()

    def findWorkoutByTime(self, time):
        time = pd.Timestamp(time, tz=pytz.FixedOffset(-240))
        print(time.tzinfo)
        print(self.pivot_workout_df.index.tzinfo)
        return self.pivot_workout_df.index[self.pivot_workout_df.index.get_loc(time, method='nearest')].strftime('%Y-%m-%d %H:%M:%S')

    def findRecordByTime(self, time):
        time = pd.Timestamp(time, tz=pytz.FixedOffset(-240))
        print(time.tzinfo)
        print(self.pivot_record_df.index.tzinfo)
        return self.pivot_record_df.index[self.pivot_record_df.index.get_loc(time, method='nearest')].strftime('%Y-%m-%d %H:%M:%S')
