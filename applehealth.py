import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import time

class AppleHealth:
    def __init__(self, fname = 'export.xml', pivotIndex = 'endDate'):
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
        print(self.pivot_workout_df)
        # print(self.pivot_workout_df.xs('Soccer'))
        print()
        e = time.time()
        print("pivot Time = {}".format(e-s))

    def resampleRecords(self, resampDict, period = 'D'):
        self.record_df = self.pivot_record_df.resample(period).agg(resampDict)

    def resampleWorkouts(self, resampDict, period = 'D'):
        self.workout_df = self.pivot_workout_df.resample(period).agg(resampDict)

    def delta(self, deltaCol):
        self.record_df['Delta'] = df[deltaCol].diff()

    def timeSeries(self, timeSeriesCol, col = 'purple', lw = 1, fsx = 12, fsy = 4):
        fig = plt.figure(figsize=(fsx,fsy))
        sns.lineplot(data=self.record_df[timeSeriesCol], color=col, linewidth=lw)
        plt.show()

    def workoutTrends(self, workout_type, metrics = ['duration', 'totalDistance', 'totalEnergyBurned'], col = 'purple', lw = 1, fsx = 12, fsy = 4):
        fig = plt.figure(figsize=(fsx,fsy))
        sns.lineplot(data=self.pivot_workout_df.loc[:,(metrics, workout_type)], color=col, linewidth=lw)
        plt.show()

    def correlationMatrix(self):
        # correlation matrix
        self.cm = self.record_df.corr()
        return self.cm

    def heatMap(self, fsx = 8, fsy = 6):
        # correlation matrix
        self.cm = self.record_df.corr()
        # heatmap
        fig = plt.figure(figsize=(fsx, fsy))
        sns.heatmap(self.cm, annot=True, fmt=".2f", vmin=-1.0, vmax=+1.0, cmap='Spectral')
        plt.show()

    def pairPlot(self,pairPlotArgs, fsx = 12, fsy = 12, col = 'purple', lw = 1,):
        fig = plt.figure(figsize=(fsx, fsy))

        g = sns.pairplot(self.record_df[pairPlotArgs],
                     kind='kde',
                     plot_kws=dict(fill=False, color=col, linewidths=lw),
                     diag_kws=dict(fill=False, color=col, linewidth=lw))

        # add observation dots
        g.map_offdiag(sns.scatterplot, marker='.', color='black')
        plt.show()

    def narrowRange(self, startDate, endDate):
        return self.pivot_record_df[startDate:endDate]

    def dayOnDay(self, metric):
        self.by_day = self.pivot_record_df[metric].resample('D').sum()
        return self.by_day

    def means_by_month(self, metric):
        means_by_distinct_month = self.dayOnDay(metric).resample('M').mean()
        means_by_month = means_by_distinct_month.groupby(means_by_distinct_month.index.month).mean()
        means_by_month.index = list(calendar.month_name)[1:]
        return means_by_month

    def monthlyBar(self, metric):
        self.means_by_month(metric).plot(kind='bar')
        plt.show()
