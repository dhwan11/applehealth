from applehealth import AppleHealth
import numpy as np
import pandas as pd
ah = AppleHealth(readCache = True)
print(ah.pivot_workout_df.columns)
# ah.resampleWorkouts({'duration' : np.mean})
# ah.workoutTrends('Soccer')
ah.resampleRecords({'BodyMass' : np.mean, 'DistanceWalkingRunning' : sum, 'StepCount' : sum, 'HeartRateVariabilitySDNN': np.mean, 'RestingHeartRate' : np.mean, 'AppleExerciseTime' : sum, 'SleepAnalysis': sum})
# print(ah.pivot_record_df['BodyMass'])
# print(ah.pivot_record_df['BodyMass'].name)
# out = ah.compareTimeFrames('2021-06-01 00:00:00-04:00', '2021-06-30 00:00:00-04:00', '2021-05-01 00:00:00-04:00', '2021-05-31 00:00:00-04:00')
# print(out[0])
# print(out[1])
# test = ah.narrowRange('2021-06-05 00:00:00-04:00', '2021-06-05 23:59:59-04:00')
# print(test)
# print(len(test))
# print(list(ah.pivot_record_df.columns))
# ah.resampleWorkouts({'BodyMass' : np.mean, 'DistanceWalkingRunning' : sum, 'StepCount' : sum, 'HeartRateVariabilitySDNN': np.mean, 'RestingHeartRate' : np.mean, 'AppleExerciseTime' : sum})
# ah.extractWorkoutType('Soccer', 'soccer.csv')
# ah.sleepRecord()
# ah.sleepAnalysis()
# print(len(ah.pivot_record_df))
# print(ah.standardDeviationRecord())
# print(ah.standardDeviationWorkout())
# print(ah.workoutAvgEnergyBurnedPerMin('TraditionalStrengthTraining'))
# print(ah.workoutAvgDistancePerMin('Soccer'))
# print(ah.workoutAvgDistancePerMin('Swimming'))
# print(ah.workoutAvgDistancePerMin('Running'))
# print(len(ah.compressRecord('5T')))
# print(ah.avgRecordFreq())
# print(ah.record_df.isna().sum()/len(ah.record_df)*100)
# print(ah.pivot_record_df.isna().sum()/len(ah.pivot_record_df)*100)
# print(len(ah.compressRecord('5T')))
# temp = ah.dropNullRecord(thresh = 0.000001)
# print(temp.isna().sum()/len(temp)*100)
# ah.timeSeries('DistanceWalkingRunning')
# print(ah.describeRecords())
# ah.recordSummaryToCsv('records summary.csv')
# print(ah.describeWorkouts())
# ah.workoutSummaryToCsv('workout summary.csv')
# ah.recordSummaryToCsv('workout summary.csv')
# print(ah.zScoreRecords())
# ah.heatMap()
# print(len(ah.cm))
# print(len(ah.cm.columns))
# ah.pairPlot(['DistanceWalkingRunning', 'AppleExerciseTime', 'SleepAnalysis'])
# ah.dropOutliers(0.0000025)
# print(ah.zScoreRecords())
# ah.heatMap()
# ah.pairPlot(['DistanceWalkingRunning', 'AppleExerciseTime', 'SleepAnalysis'])
# print(ah.narrowRange('2021-06-05 00:00:00-04:00', '2021-06-08 00:00:00-04:00'))
# print(ah.monthlyBar('DistanceWalkingRunning'))
# key = ah.findWorkoutByTime('2019-6-19 17:00:0')
# print(key)
# print(ah.pivot_workout_df.loc[key])
# print(ah.pivot_record_df.loc[key])

# print(ah.bestWorkouts('Soccer', 10, 'totalEnergyBurned'))
# print(ah.labelRecord('DistanceWalkingRunning', ['S', 'M', 'L'], [[0, 2],[2, 4],[4, 6]]))
# print(ah.labelWorkout('totalEnergyBurned', 'Soccer', ['S', 'M', 'L'], [[0, 250], [250, 750], [750, 1500]]))

# wear time and how to define it
# comparision between lower and higher level of data
# label ongoing activity
# karvonen form hr -> activity intensity
# step count intensity
# make additional features based on lit
# entropy of hr
# corrleation between step count and hr
# make defaults sensible and reasonable and tranparent
# study style and multiple exports