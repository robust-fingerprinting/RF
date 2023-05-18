from RF.const_rf import *

def fun(times, sizes):
    feature = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            if times[i] >= maximum_load_time:
                feature[0][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
        if sizes[i] < 0:
            if times[i] >= maximum_load_time:
                feature[1][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += 1

    return feature
