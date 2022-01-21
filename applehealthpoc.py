from applehealth import AppleHealth
import numpy as np

ah = AppleHealth()
# ah.resampleWorkouts({'duration' : np.mean})
ah.workoutTrends('Soccer')
ah.resampleRecords({'BodyMass' : np.mean, 'DistanceWalkingRunning' : sum, 'StepCount' : sum, 'HeartRateVariabilitySDNN': np.mean, 'RestingHeartRate' : np.mean, 'AppleExerciseTime' : sum})
# ah.resampleWorkouts({'BodyMass' : np.mean, 'DistanceWalkingRunning' : sum, 'StepCount' : sum, 'HeartRateVariabilitySDNN': np.mean, 'RestingHeartRate' : np.mean, 'AppleExerciseTime' : sum})
print(ah.record_df)
ah.timeSeries('DistanceWalkingRunning')
ah.heatMap()
ah.pairPlot(['DistanceWalkingRunning', 'StepCount', 'HeartRateVariabilitySDNN', 'RestingHeartRate', 'AppleExerciseTime'])
print(ah.narrowRange('2021-06-05 00:00:00-04:00', '2021-06-08 00:00:00-04:00'))
print(ah.monthlyBar('DistanceWalkingRunning'))
