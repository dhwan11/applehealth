from applehealth import AppleHealth
import numpy as np
import pandas as pd
import tracemalloc
import sys
import os
# tracemalloc.start()
runtime = []
dfSize = []
# ah = AppleHealth(readCache=True)
# dfSize.append(ah.pivot_record_df.memory_usage(deep=True).sum() + ah.pivot_workout_df.memory_usage(deep=True).sum() if ah.pivot_workout_df is not None else ah.pivot_record_df.memory_usage(deep=True).sum())
# file_size = os.path.getsize('export.xml')
# runtime.append({ah.runtime, file_size})
file_size = os.path.getsize('export.xml'.format(0))
for n in range(0, 10):
    print(n)
    ah = AppleHealth('export.xml'.format(0))
    dfSize.append(ah.pivot_record_df.memory_usage(deep=True).sum() + ah.pivot_workout_df.memory_usage(deep=True).sum() if ah.pivot_workout_df is not None else ah.pivot_record_df.memory_usage(deep=True).sum())
    # file_size = os.path.getsize('done28\export.{:02d}.xml'.format(n))
    runtime.append(ah.runtime)
print(sum(runtime)/len(runtime))
print(sum(dfSize)/len(dfSize))
print(file_size)
# print(tracemalloc.get_traced_memory())
# # print(sys.getsizeof(ah.pivot_record_df))
# tracemalloc.stop()