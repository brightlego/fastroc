import fastroc
import numpy as np
import time

p = np.random.random((300, 360, 40))

y_true = p < 0.2
y_score = np.random.random((300, 360, 40))/10
y_score[y_true] += 0.05
print("Starting imprecise test")
start = time.time_ns()
roc_auc = fastroc.calc_roc_auc(y_true=y_true, y_score=y_score, thread_count=16, axis=2, integral_precision=50)
end = time.time_ns()

print(f"{(end-start)/1000000:.3f}ms")

print("Starting precice test")
start = time.time_ns()
roc_auc2 = fastroc.calc_roc_auc(y_true=y_true, y_score=y_score, thread_count=16, axis=2, integral_precision=0)
end = time.time_ns()

print(f"{(end-start)/1000000:.3f}ms")