import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import time
from datetime import datetime
import pytz
from scipy import stats
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
        e = time.time()
        print("rest Time = {}".format(e-s))

        # pivot
        s = time.time()
        self.pivot_record_df = self.record_data.pivot_table(index='endDate', columns='type', values='value')
        self.pivot_workout_df = self.workout_data.pivot_table(index='endDate', columns='workoutActivityType', values=['duration', 'totalDistance', 'totalEnergyBurned'])
        e = time.time()
        print("pivot Time = {}".format(e-s))

        if writeCache:
            self.cacheAll()

    # resample the record dataframe to period and perform calculations to metrics listed in dict
    def resampleRecords(self, resampDict, period = 'D'):
        self.pivot_record_df = self.pivot_record_df.resample(period).agg(resampDict)
        return self.pivot_record_df

    # resample the workout dataframe to period and perform calculations to metrics listed in dict
    def resampleWorkouts(self, resampDict, period = 'D'):
        self.pivot_workout_df = self.pivot_workout_df.resample(period).agg(resampDict)
        return self.pivot_workout_df

    # calculate the delta of deltaCol
    # TODO: Test
    # def delta(self, deltaCol):
    #     self.record_df['Delta'] = df[deltaCol].diff()

    # display a time time series of the provided timeSeriesCol
    # TODO: allow for selecting of range of dates
    def timeSeries(self, timeSeriesCol, col = 'purple', lw = 1, fsx = 12, fsy = 4):
        fig = plt.figure(figsize=(fsx,fsy))
        sns.lineplot(data=self.pivot_record_df[timeSeriesCol], color=col, linewidth=lw)
        plt.show()

    # display workout trends of selected workout ployyed with metrics
    def workoutTrends(self, workout_type, metrics = ['duration', 'totalDistance', 'totalEnergyBurned'], col = 'purple', lw = 1, fsx = 12, fsy = 4):
        fig = plt.figure(figsize=(fsx,fsy))
        sns.lineplot(data=self.pivot_workout_df.loc[:,(metrics, workout_type)], color=col, linewidth=lw)
        plt.show()

    # returns a correlation matrix of the record dataframe
    def correlationMatrix(self):
        # correlation matrix
        self.cm = self.pivot_record_df.corr()
        return self.cm

    # displays the correlation matrix cm as a heat map
    def heatMap(self, fsx = 8, fsy = 6):
        # correlation matrix
        self.cm = self.pivot_record_df.corr()
        # heatmap
        fig = plt.figure(figsize=(fsx, fsy))
        sns.heatmap(self.cm, annot=True, fmt=".2f", vmin=-1.0, vmax=+1.0, cmap='Spectral')
        plt.show()

    # displays a pair plot of the record dataframe
    def pairPlot(self, pairPlotArgs, fsx = 12, fsy = 12, col = 'purple', lw = 1,):
        fig = plt.figure(figsize=(fsx, fsy))

        g = sns.pairplot(self.pivot_record_df[pairPlotArgs],
                     kind='kde',
                     plot_kws=dict(fill=False, color=col, linewidths=lw),
                     diag_kws=dict(fill=False, color=col, linewidth=lw),
                     dropna=True)

        # add observation dots
        g.map_offdiag(sns.scatterplot, marker='.', color='black')
        plt.show()

    # returns a slice of the pivot record data frame
    def narrowRange(self, startDate, endDate):
        return self.pivot_record_df[startDate:endDate]

    # returns the resampled chosen metric of the pivot record dataframe to a daily sum
    def dayOnDay(self, metric):
        by_day = self.pivot_record_df[metric].resample('D').sum()
        return by_day

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
    def extractWorkoutTypeToCsv(self, workout_type, fname, metrics = ['duration', 'totalDistance', 'totalEnergyBurned']):
        self.pivot_workout_df.loc[:,(metrics, workout_type)].to_csv(fname)

    # returns a dataframe of workout_type and selected metrics
    def extractWorkoutType(self, workout_type, metrics = ['duration', 'totalDistance', 'totalEnergyBurned']):
        return self.pivot_workout_df.loc[:,(metrics, workout_type)]

    # writes all all workouts to a csv file fname
    def extractAllWorkoutToCsv(self, fname):
        self.pivot_workout_df.to_csv(fname)

    # writes all all records to a csv file fname
    def extractAllRecordsToCsv(self, fname):
        self.pivot_record_df.to_csv(fname)

    # writes all specific records of record_type to a csv file fname
    def extractRecordTypeToCsv(self, record_type, fname):
        self.pivot_record_df[record_type].to_csv(fname)

    # returns all specific records of record_type
    def extractRecordType(self, record_type):
        return self.pivot_record_df[record_type]

    # generates sleep dataframe to store sleep analysis data
    def sleepRecord(self):
        self.sleep_df = self.pivot_record_df['SleepAnalysis']

    def dropNullSleep(self, subset = None, thresh = 0.0, axis = 0, inplace=False):
        return self.sleep_df.dropna(thresh  = int(thresh * len(self.sleep_df.index)), axis = axis , inplace = inplace, subset = subset)


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
        # print(int( 0.000001 * self.pivot_record_df.shape[0]))
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

    # evaluate accuracy
    def dropOutliers(self, thresh = 3):
        self.pivot_record_df = self.pivot_record_df[(np.abs(stats.zscore(self.pivot_record_df, nan_policy='omit')) < thresh).all(axis=1)]

    # evaluate use case
    def zScoreRecords(self):
        return (np.abs(stats.zscore(self.pivot_record_df.all(axis=1), nan_policy='omit')))

    def describeRecords(self):
        return self.pivot_record_df.describe()

    def recordSummaryToCsv(self, fname):
        self.pivot_record_df.describe().to_csv(fname)

    def workoutSummaryToCsv(self, fname):
        self.pivot_workout_df.describe().to_csv(fname)

    def describeWorkouts(self):
        return self.pivot_workout_df.describe()

    def compareTimeFrames(self, startDate1, endDate1, startDate2, endDate2):
        return [self.narrowRange(startDate1, endDate1).describe(), self.narrowRange(startDate2, endDate2).describe()]
    
    def bestWorkouts(self, workout_type, by, k = 10):
        """ Finds the k best workouts of workout type

        Args:
            workout_type (String): Name/Title of workout
            by (String): Metric to sort the worouts by
            k (int, optional): number of workouts to return. Defaults to 10.

        Returns:
            Pd.Series: A pandas series of the best workouts
        """
        return self.pivot_workout_df[by][workout_type].sort_values(ascending=False)[:k]
    
    def worstWorkouts(self, workout_type, by, k = 10):
        """ Finds the k worst workouts of workout type

        Args:
            workout_type (String): Name/Title of workout
            by (String): Metric to sort the worouts by
            k (int, optional): number of workouts to return. Defaults to 10.

        Returns:
            Pd.Series: A pandas series of the worst workouts
        """
        return self.pivot_workout_df[by][workout_type].sort_values()[:k]
        

    def labelRecord(self, record_type, labels, thresholds):
        assert len(labels) == len(thresholds)
        new = self.pivot_record_df[record_type].to_frame().assign(label='None').reset_index(drop=True)
        for i in range(len(labels)): 
            elemsIdx = np.where(np.logical_and(self.pivot_record_df[record_type] > thresholds[i][0], self.pivot_record_df[record_type] < thresholds[i][1]))
            for ei in elemsIdx:
                new.at[ei, 'label'] = labels[i]
        return new       
    
    def labelWorkout(self, by, workout_type, labels, thresholds):
        assert len(labels) == len(thresholds)
        new = self.pivot_workout_df[by][workout_type].to_frame().assign(label='None').reset_index(drop=True)
        for i in range(len(labels)): 
            elemsIdx = np.where(np.logical_and(self.pivot_workout_df[by][workout_type] > thresholds[i][0], self.pivot_workout_df[by][workout_type] < thresholds[i][1]))
            for ei in elemsIdx:
                new.at[ei, 'label'] = labels[i]
        return new 
    
    def calculateDaysBetweenDates(self, begin, end):
        return (end - begin).days
    
    def findRecordbyDate(self, date):
        return self.pivot_record_df[self.pivot_record_df.index[self.pivot_record_df.index.get_loc(date, method='nearest')].strftime('%Y-%m-%d')]
    
    def findWorkoutbyDate(self, date):
        return self.pivot_workout_df[self.pivot_workout_df.index[self.pivot_workout_df.index.get_loc(date, method='nearest')].strftime('%Y-%m-%d')]
    
    def findBestWorkoutsbyDate(self, date, k = 10):
        return self.findWorkoutbyDate(date).sort_values(by=['totalEnergyBurned', 'totalDistance'], ascending=False)[:k]
    
    def findWorstWorkoutsbyDate(self, date, k = 10):
        return self.findWorkoutbyDate(date).sort_values(by=['totalEnergyBurned', 'totalDistance'])[:k]
    
    def findBestRecordsbyDate(self, date, k = 10):
        return self.findRecordbyDate(date).sort_values(by=['totalEnergyBurned', 'totalDistance'], ascending=False)[:k]
    
    def findWorstRecordsbyDate(self, date, k = 10):
        return self.findRecordbyDate(date).sort_values(by=['totalEnergyBurned', 'totalDistance'])[:k]
    
    def findBestWorkoutsbyTime(self, time, k = 10):
        return self.findWorkoutbyTime(time).sort_values(by=['totalEnergyBurned', 'totalDistance'], ascending=False)[:k]
    
    def findWorstWorkoutsbyTime(self, time, k = 10):
        return self.findWorkoutbyTime(time).sort_values(by=['totalEnergyBurned', 'totalDistance'])[:k]
    
    def calculateWorkoutDuration(self, workout_type):
        return self.pivot_workout_df['duration'][workout_type].sum()
    
    def calculateRecordDuration(self, record_type):
        return self.pivot_record_df['duration'][record_type].sum()
    
    def calculateWorkoutDistance(self, workout_type):
        return self.pivot_workout_df['totalDistance'][workout_type].sum()
    
    def calculateRecordDistance(self, record_type):
        return self.pivot_record_df['totalDistance'][record_type].sum()
    
    def dailyWorkoutSummary(self, workout_type):
        return self.pivot_workout_df['totalEnergyBurned'][workout_type].resample('D').sum()
    
    def dailyRecordSummary(self, record_type):
        return self.pivot_record_df['totalEnergyBurned'][record_type].resample('D').sum()
    
    def returnWorkoutSummary(self, workout_type):
        return self.pivot_workout_df['totalEnergyBurned'][workout_type].describe()
    
    def barGraphWorkoutSummary(self, workout_type):
        return self.pivot_workout_df['totalEnergyBurned'][workout_type].describe().to_frame().T
    
    def timeSeriesWorkoutSummary(self, workout_type):
        return self.pivot_workout_df['totalEnergyBurned'][workout_type].resample('D').sum()
    
    def plotWorkoutSummary(self, workout_type):
        self.pivot_workout_df['totalEnergyBurned'][workout_type].resample('D').sum().plot(kind='bar')
        plt.show()
        
    def plotDailyDistance(self, workout_type):
        self.pivot_workout_df['totalDistance'][workout_type].resample('D').sum().plot(kind='bar')
        plt.show()
        
    def plotDailyDuration(self, workout_type):
        self.pivot_workout_df['duration'][workout_type].resample('D').sum().plot(kind='bar')
        plt.show()
        
    def calculateDailyDuration(self, workout_type):
        return self.pivot_workout_df['duration'][workout_type].resample('D').sum()
    
    def probabilityDistribution(self, workout_type, by):
        return self.pivot_workout_df[by][workout_type].value_counts(normalize=True)
    
    def binomialProbabilityDistribution(self, workout_type, by, n):
        return self.pivot_workout_df[by][workout_type].value_counts(normalize=True).cumsum()
    
    def predictWorkout(self, workout_type, by, n):
        return self.pivot_workout_df[by][workout_type].value_counts(normalize=True).cumsum().index[n]
    
    def predictRecord(self, record_type, by, n):
        return self.pivot_record_df[by][record_type].value_counts(normalize=True).cumsum().index[n]
    
    def predictDaybasedOnWorkout(self, workout_type, by, n):
        return self.pivot_workout_df[by][workout_type].value_counts(normalize=True).cumsum().index[n]
    
    def sortWorkoutsbyDuration(self, workout_type):
        return self.pivot_workout_df['duration'][workout_type].sort_values(ascending=False)
    
    def sortWorkoutsbyEnergy(self, workout_type):
        return self.pivot_workout_df['totalEnergyBurned'][workout_type].sort_values(ascending=False)
    
    def sortWorkoutsbyDistance(self, workout_type):
        return self.pivot_workout_df['totalDistance'][workout_type].sort_values(ascending=False)
    
    def sortSleepbyDuration(self, sleep_type):
        return self.pivot_sleep_df['duration'][sleep_type].sort_values(ascending=False)
    
    def createSleepSummary(self, sleep_type):
        return self.pivot_sleep_df['totalEnergyBurned'][sleep_type].describe()
    
    def compareworkouts(self, workout_type1, workout_type2):
        return self.pivot_workout_df['totalEnergyBurned'][workout_type1].describe().to_frame().T.join(self.pivot_workout_df['totalEnergyBurned'][workout_type2].describe().to_frame().T)
    
    def pieChart(self, workout_type):
        return self.pivot_workout_df['totalEnergyBurned'][workout_type].value_counts(normalize=True).plot(kind='pie', autopct='%1.1f%%')