import fastroc
import numpy as np
import time

p = np.random.random((300, 360, 40))

y_true = p < 0.2
y_score = np.random.random((300, 360, 40))/10
y_score[y_true] += 0.05

y_true = np.array(y_true, dtype=np.int64)
print(y_score)
y_score = np.array(y_score, dtype=np.float32)
print(y_score)

start = time.time_ns()
roc_auc = fastroc.calc_roc_auc(y_true=y_true, y_score=y_score, thread_count=16, axis=2)
end = time.time_ns()

print(f"{(end-start)/1000000:.3f}ms")
