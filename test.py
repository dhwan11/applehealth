from datetime import timedelta
from sqlite3 import Time
import unittest

from pandas import TimedeltaIndex
from applehealth import AppleHealth
import numpy as np
import time
class TestSum(unittest.TestCase):

    def setUp(self):
        self.ah = AppleHealth(readCache = True)

    def testResampling(self):
        self.ah.resampleRecords({'BodyMass' : np.mean, 'DistanceWalkingRunning' : sum, 'StepCount' : sum, 'HeartRateVariabilitySDNN': np.mean, 'RestingHeartRate' : np.mean, 'AppleExerciseTime' : sum, 'SleepAnalysis': sum})
        assert list(self.ah.pivot_record_df.columns) == ['BodyMass', 'DistanceWalkingRunning', 'StepCount', 'HeartRateVariabilitySDNN', 'RestingHeartRate', 'AppleExerciseTime', 'SleepAnalysis']

    def testTimeSeries(self):
        self.ah.resampleRecords({'BodyMass': np.mean})
        assert self.ah.pivot_record_df['BodyMass'].empty == False
        assert self.ah.pivot_record_df['BodyMass'].name == 'BodyMass'

    def testWorkoutTrends(self):
        metrics = ['duration', 'totalDistance', 'totalEnergyBurned']
        workout_type = 'Soccer'
        assert self.ah.pivot_workout_df.loc[:,(metrics, workout_type)].empty == False
        l = list(map(lambda x: x[0], self.ah.pivot_workout_df.loc[:,(metrics, workout_type)].columns))
        assert l == metrics

    def testNarrowRange(self):
        self.ah.resampleRecords({'BodyMass' : np.mean, 'DistanceWalkingRunning' : sum, 'StepCount' : sum, 'HeartRateVariabilitySDNN': np.mean, 'RestingHeartRate' : np.mean, 'AppleExerciseTime' : sum, 'SleepAnalysis': sum})
        test = self.ah.narrowRange('2021-06-05 00:00:00-04:00', '2021-06-05 23:59:59-04:00')
        assert len(test) == 1
        assert test.size == 7

    def testCompareTimeFrames(self):
        self.ah.resampleRecords({'BodyMass' : np.mean, 'DistanceWalkingRunning' : sum, 'StepCount' : sum, 'HeartRateVariabilitySDNN': np.mean, 'RestingHeartRate' : np.mean, 'AppleExerciseTime' : sum, 'SleepAnalysis': sum})
        out = self.ah.compareTimeFrames('2021-06-01 00:00:00-04:00', '2021-06-01 23:59:59-04:00', '2021-05-01 00:00:00-04:00', '2021-05-01 23:59:59-04:00')
        assert len(out[0]) == 8
        assert out[0].size == 56
        assert len(out[0].columns) == 7
        assert len(out[1]) == 8
        assert out[1].size == 56
        assert len(out[1].columns) == 7
        
    def testBestWorkouts(self):
        out = self.ah.bestWorkouts('Soccer', 'totalEnergyBurned', 10)
        assert len(out) == 10
        prevMax = 9999999999
        for w in out:
            assert w < prevMax
            prevMax = w
        
    def testWorstWorkouts(self):
        out = self.ah.worstWorkouts('Soccer', 'totalEnergyBurned', 10)
        assert len(out) == 10
        prevMin = 0
        for w in out:
            assert w > prevMin
            prevMin = w
            
            
    def testCM(self):
        self.ah.resampleRecords({'BodyMass' : np.mean, 'DistanceWalkingRunning' : sum, 'StepCount' : sum, 'HeartRateVariabilitySDNN': np.mean, 'RestingHeartRate' : np.mean, 'AppleExerciseTime' : sum, 'SleepAnalysis': sum})
        out = self.ah.correlationMatrix()
        assert out.equals(self.ah.cm)
        assert len(out) == 7
        
    def testDayOnDay(self):
        out = self.ah.dayOnDay('DistanceWalkingRunning')
        assert out.name == 'DistanceWalkingRunning'
        
    def testFreq(self):
        oldSize = len(self.ah.pivot_record_df)
        oldFreq = self.ah.avgRecordFreq()
        self.ah.pivot_record_df = self.ah.pivot_record_df.resample('D').mean()
        newSize = len(self.ah.pivot_record_df)
        newFreq = self.ah.avgRecordFreq()
        assert newSize < oldSize
        assert newFreq > oldFreq
        
    def testExtractWorkoutType(self):
        out = self.ah.extractWorkoutType('Soccer')
        assert out.columns[0][1] == 'Soccer'
        assert out.columns[0][0] == 'duration'
        assert len(out) == 775
        assert len(out.columns) == 3
        
    def testExtractRecordType(self):
        out = self.ah.extractRecordType('DistanceWalkingRunning')
        assert  out.name == 'DistanceWalkingRunning'
        assert len(out) == 1134761
    
    def testSleepRecord(self):
        # assert  self.ah.sleep_df == None
        self.ah.sleepRecord()
        assert self.ah.sleep_df.empty == False
        
    def testCompressRecord(self):
        out = self.ah.compressRecord()
        assert len(out) < len(self.ah.pivot_record_df)
        
    def testAvgRecordFreq(self):
        self.assertIsInstance(self.ah.avgRecordFreq(), timedelta)
        
    def testNonCache(self):
        test = AppleHealth()
        assert self.ah.pivot_record_df.equals(test.pivot_record_df)
        
    def testWriteCache(self):
        AppleHealth(readCache=True, writeCache=True)

if __name__ == '__main__':
    unittest.main()
