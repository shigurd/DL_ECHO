import csv
import statistics

def calculate_max_diff_for_views(view_diam_list):
    true_list_cm = []
    pred_list_cm = []

    for i, n in enumerate(view_diam_list):
        for k in n:
           if k == 'true_diam_cm':
               true_list_cm.append(float(view_diam_list[i][k]))
           elif k == 'pred_diam_cm':
               pred_list_cm.append(float(view_diam_list[i][k]))
    true_list_cm.sort()
    pred_list_cm.sort()

    true_max_diff_cm = true_list_cm[-1] - true_list_cm[0]
    pred_max_diff_cm = pred_list_cm[-1] - pred_list_cm[0]

    return true_max_diff_cm, pred_max_diff_cm


def calculate_intervariability_per_patient(csv_input, writer):
    lvot_patient_sorted_dict = dict()

    with open(csv_input, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        ''' skips header '''
        next(csv_reader, None)

        for row in csv_reader:
            fn, measure_type, view_type, img_quality, gt_quality, pred_diam_cm, true_diam_cm, diff_diam_cm, i_ed_cm, s_ed_cm, tot_ed_cm, i_x_diff_cm, i_y_diff_cm, s_x_diff_cm, s_y_diff_cm, pred_diam_pix, true_diam_pix, diff_diam_pix, i_ed_pix, s_ed_pix, tot_ed_pix, i_x_diff_pix, i_y_diff_pix, s_x_diff_pix, s_y_diff_pix = row #sjekke dette
            patient_id = fn.split('_', 1)[0]

            if patient_id not in lvot_patient_sorted_dict:
                lvot_patient_sorted_dict[patient_id] = dict()
                lvot_patient_sorted_dict[patient_id][fn] = dict()
            else:
                lvot_patient_sorted_dict[patient_id][fn] = dict()

            lvot_patient_sorted_dict[patient_id][fn]['view_type'] = view_type
            lvot_patient_sorted_dict[patient_id][fn]['true_diam_cm'] = true_diam_cm
            lvot_patient_sorted_dict[patient_id][fn]['true_diam_pix'] = true_diam_pix
            lvot_patient_sorted_dict[patient_id][fn]['pred_diam_cm'] = pred_diam_cm
            lvot_patient_sorted_dict[patient_id][fn]['pred_diam_pix'] = pred_diam_pix
            lvot_patient_sorted_dict[patient_id][fn]['img_quality'] = img_quality
            lvot_patient_sorted_dict[patient_id][fn]['gt_quality'] = gt_quality

    list_plax_pred = []
    list_plax_true = []
    list_zoom_pred = []
    list_zoom_true = []
    list_all_true = []
    list_all_pred = []

    for x in lvot_patient_sorted_dict:
        if len(lvot_patient_sorted_dict[x]) > 1:
            print(x)
            print(len(lvot_patient_sorted_dict[x]))

            plax_per_patient = []
            zoom_per_patient = []
            all_per_patient = []

            for y in lvot_patient_sorted_dict[x]:
                #if lvot_patient_sorted_dict[x][y]['img_quality'] != 'ILOW' and lvot_patient_sorted_dict[x][y]['gt_quality'] != 'MLOW': #for filtering HMHM
                if lvot_patient_sorted_dict[x][y]['view_type'] == 'PLAX':
                    plax_per_patient.append(lvot_patient_sorted_dict[x][y])
                    all_per_patient.append(lvot_patient_sorted_dict[x][y])
                elif lvot_patient_sorted_dict[x][y]['view_type'] == 'ZOOM':
                    zoom_per_patient.append(lvot_patient_sorted_dict[x][y])
                    all_per_patient.append(lvot_patient_sorted_dict[x][y])

            if len(plax_per_patient) > 1:
                max_diff_plax = calculate_max_diff_for_views(plax_per_patient)
                list_plax_true.append(max_diff_plax[0])
                list_plax_pred.append(max_diff_plax[1])
            else:
                max_diff_plax = ['-', '-']

            if len(zoom_per_patient) > 1:
                max_diff_zoom = calculate_max_diff_for_views(zoom_per_patient)
                list_zoom_true.append(max_diff_zoom[0])
                list_zoom_pred.append(max_diff_zoom[1])
            else:
                max_diff_zoom = ['-', '-']

            if len(all_per_patient) > 1:
                max_diff_all = calculate_max_diff_for_views(all_per_patient)
                list_all_true.append(max_diff_all[0])
                list_all_pred.append(max_diff_all[1])
            else:
                max_diff_all = ['-', '-']

            if max_diff_all != ['-', '-'] or max_diff_zoom != ['-', '-'] or max_diff_plax != ['-', '-']:
                writer.writerow([x, max_diff_all[0], max_diff_all[1], len(all_per_patient), max_diff_plax[0], max_diff_plax[1], len(plax_per_patient), max_diff_zoom[0], max_diff_zoom[1], len(zoom_per_patient)])

            print('max_diff_plax', max_diff_plax)
            print('max_diff_zoom', max_diff_zoom)
            print('max_diff_all', max_diff_all)

    print('plax', len(list_plax_true))
    print('zoom', len(list_zoom_true))
    print('all', len(list_all_true))

    print('mean_plax_pred', sum(list_plax_pred)/len(list_plax_pred))
    print('mean_plax_true', sum(list_plax_true)/len(list_plax_true))
    print('mean_zoom_pred', sum(list_zoom_pred)/len(list_zoom_pred))
    print('mean_zoom_true', sum(list_zoom_true)/len(list_zoom_true))
    print('mean_all_pred', sum(list_all_pred)/len(list_all_pred))
    print('mean_all_true', sum(list_all_true)/len(list_all_true))

    print('median_plax_pred', statistics.median(list_plax_pred))
    print('median_plax_true', statistics.median(list_plax_true))
    print('median_zoom_pred', statistics.median(list_zoom_pred))
    print('median_zoom_true', statistics.median(list_zoom_true))
    print('median_all_pred', statistics.median(list_all_pred))
    print('median_all_true', statistics.median(list_all_true))


if __name__ == '__main__':
    input_csv = r'D:\DL_ECHO\LVOT_point_detection\predictions\final\GE1408_HMLHMLAVA___Mar11_14-43-45_EFFIB2UNETIMGN_DSNT_ADAM_LR5_AL_T-GE1408_HMLHMLAVA_V-NONE_EP30_LR0.003_BS32_VAL\COORD_DATA.txt'
    output_csv = 'max_measurement_difference_patient_HMLHML.csv'
    data_logger = open(output_csv, 'w', newline='')
    writer = csv.writer(data_logger)
    writer.writerow(['patient','all_true','all_pred','n_all','plax_true','plax_pred','n_plax','zoom_true','zoom_pred','n_zoom'])

    calculate_intervariability_per_patient(input_csv, writer)
    data_logger.close()