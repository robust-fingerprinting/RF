# coding: utf-8
import numpy as np
import csv

eps = 1e-6


def pre_recCall(readFile, writeFile, n):
    import csv
    tp = {}
    tp_fn = {}
    tp_fp = {}
    Tp = 0
    Fn = 0
    all_l = 0
    total = [[0] * n for i in range(n)]
    csvwritefile = open(writeFile, "w", newline='')
    fieldnames = ['appName', 'precision', 'recall', 'Weight', 'number', 'tp']
    writer = csv.DictWriter(csvwritefile, delimiter=",", fieldnames=fieldnames)
    writer.writerow({'appName': 'appName', 'precision': 'precision', 'recall': 'recall',
                     'Weight': 'Weight', 'number': 'number', 'tp': 'tp'})
    csvreadfile = open(readFile, "r")
    reader = csv.reader(csvreadfile, delimiter=",")
    for real, classres in reader:
        total[int(real)][int(classres)] += 1
        if not tp_fn.__contains__(real):
            tp_fn[real] = 0
            tp[real] = 0
        if not tp_fp.__contains__(real):
            tp_fp[real] = 0
        if not tp_fp.__contains__(classres):
            tp_fp[classres] = 0
        if real == classres:
            tp[real] += 1
            Tp += 1
        else:
            Fn += 1
        tp_fn[real] += 1
        tp_fp[classres] += 1
        all_l += 1
    csvreadfile.close()
    for key in tp_fn:
        try:
            preci = 1.0 * tp[key] / tp_fp[key]
        except:
            preci = 0
        recall = 1.0 * tp[key] / tp_fn[key]
        weight = 1.0 * tp_fn[key] / all_l
        writer.writerow({'appName': key, 'precision': '{:.4f}'.format(preci), 'recall': '{:.4f}'.format(recall),
                         'Weight': '{:.4f}'.format(weight), 'number': tp_fn[key], 'tp': tp[key]})
    print(Tp, all_l)
    writer.writerow({'appName': 'accuracy', 'precision': '{:.4f}'.format(Tp * 1.0 / all_l), 'recall': '',
                     'Weight': ' ', 'number': ' ', 'tp': ' '})

    row_writer = csv.writer(csvwritefile)
    row_writer.writerow([])
    row_writer.writerow([' '] + [i for i in range(n)])
    for i in range(len(total)):
        row_writer.writerow([i] + total[i])
    row_writer.writerow([])
    for i in range(n):
        info = []
        info.append(i)
        for j in range(n):
            if total[i][j] > 0:
                info.append(str(j) + '@' + str(total[i][j]))
        row_writer.writerow(info)
    csvwritefile.close()
    return Tp * 1.0 / all_l


def score_func_precision_recall(result_file, website_res, unmon_label):
    file = open(result_file, 'w+', encoding='utf-8', newline='')
    csvwirter = csv.writer(file)
    upper_bound = 1.0
    thresholds = upper_bound - upper_bound / np.logspace(0.05, 2, num=15, endpoint=True)
    csvwirter.writerow(['TH  ', 'TP   ', 'TN   ', 'FP   ', 'FN   ', 'Pre. ', 'Rec. '])
    fmt_str = '{:.2f}:\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}'

    # evaluate list performance at different thresholds
    # high threshold will yield higher precision, but reduced recall
    for TH in thresholds:
        TP, FP, TN, FN = 0, 0, 0, 0

        # Test with Monitored testing instances
        for i in range(len(website_res)):
            ground_truths = website_res[i][0]
            sm_vector = np.array(website_res[i][1:])
            predicted_class = np.argmax(sm_vector)
            max_prob = max(sm_vector)
            if ground_truths != unmon_label:
                if predicted_class == ground_truths:  # predicted as Monitored
                    if max_prob >= TH:  # predicted as Monitored and actual site is Monitored
                        TP = TP + 1
                    else:  # predicted as Unmonitored and actual site is Monitored
                        FN = FN + 1
                else:  # predicted as Unmonitored and actual site is Monitored
                    FN = FN + 1
            else:
                if predicted_class != unmon_label:  # predicted as Monitored
                    if max_prob >= TH:  # predicted as Monitored and actual site is Unmonitored
                        FP = FP + 1
                    else:  # predicted as Unmonitored and actual site is Unmonitored
                        TN = TN + 1
                else:  # predicted as Unmonitored and actual site is Unmonitored
                    TN = TN + 1
        res = [TH, TP, TN, FP, FN, float(TP) / (TP + FP + eps), float(TP) / (TP + FN + eps)]
        print(fmt_str.format(*res))
        csvwirter.writerow(res)

    file.close()
    return 'finish'
