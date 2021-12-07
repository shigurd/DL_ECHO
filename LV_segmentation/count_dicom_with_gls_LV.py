import os
import csv

class GLStracker:
    def __init__(self, unique_id):
        self.name = unique_id
        self.four_channel = 0
        self.two_channel = 0
        self.aplax = 0

    def register_projection(self, projection_string):
        if projection_string == '4CH':
            self.four_channel += 1
        elif projection_string == '2CH':
            self.two_channel += 1
        elif projection_string == 'APLAX':
            self.aplax += 1

    def get_stats(self):
        return [self.four_channel, self.two_channel, self.aplax]

    def get_name(self):
        return self.name

    def has_gls(self):
        if self.four_channel != 0 and self.two_channel != 0 and self.aplax != 0:
            return True
        else:
            return False


def csv_input_gls_count(csv_file):
    gls_list = []
    count_hcm_patients = []
    count_angio_patients = []

    with open(csv_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        header = next(csv_reader)

        for row in csv_reader:
            file_id, exam_id, projection, img_quality, mask_quality = row

            if len(exam_id.split('_')) == 3:
                patient_id = exam_id.split('_')[0]
                if patient_id not in count_hcm_patients:
                    count_hcm_patients.append(patient_id)

            else:
                patient_id = exam_id

            if exam_id[0:5] == 'ANGIO':
                if patient_id not in count_angio_patients:
                    count_angio_patients.append(patient_id)

            found = False
            for x in gls_list:
                if x.get_name() == patient_id:
                    found = True
                    x.register_projection(projection)

            if found == False:
                temp = GLStracker(patient_id)
                temp.register_projection(projection)
                gls_list.append(temp)

    checksum = 0
    gls_patients = 0
    gls_angio = 0
    gls_hcm = 0
    for e in gls_list:
        if e.has_gls():
            gls_patients += 1
            if e.get_name()[0:5] == 'ANGIO':
                gls_angio += 1
            if e.get_name()[0:3] == 'HCM':
                gls_hcm += 1

        print(e.get_name(), e.has_gls(), e.get_stats())
        checksum += sum(e.get_stats())

    #print('hcm patients n =', len(count_hcm_patients))
    #print('hcm patients gls n =', gls_hcm)
    #print('angio patients n =', len(count_angio_patients))
    #print('angio patients gls n =', gls_angio)
    print('total patients n =', len(gls_list))
    print('total patients with GLS n =', gls_patients)
    print('total exams', checksum)


def keep_gls_count(dcm_folder, csv_file):
    gls_list = []

    for fn in os.listdir(dcm_folder):
        if fn.rsplit('.', 1)[-1] == 'dcm':
            fn = fn.rsplit('.', 1)[0]

        file_id, patient_id, exam_id, projection, img_quality, mask_quality, data_setting = get_csv_tags(fn, csv_file)

        if len(exam_id.split('_')) == 3:
            patient_id = exam_id.split('_')[0]
        else:
            patient_id = exam_id

        found = False
        for x in gls_list:
            if x.get_name() == patient_id:
                found = True
                x.register_projection(projection)

        if found == False:
            temp = GLStracker(patient_id)
            temp.register_projection(projection)
            gls_list.append(temp)

    checksum = 0
    gls_patients = 0

    for e in gls_list:
        if e.has_gls():
            gls_patients += 1

        print(e.get_name(), e.has_gls(), e.get_stats())
        checksum += sum(e.get_stats())

    print(f'total patients in {dcm_folder} n =', len(gls_list))
    print(f'total patients with GLS in {dcm_folder} n =', gls_patients)
    print(f'total exams in {dcm_folder}', checksum)


def get_csv_tags(dcm_fn, csv_file):

    with open(csv_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        header = next(csv_reader)

        found = False
        for row in csv_reader:
            file_id, patient_id, exam_id, projection, img_quality, mask_quality, data_setting = row

            if file_id == dcm_fn:
                found = True
                return row

        if found == False:
            print(dcm_fn, 'not found')


if __name__ == "__main__":
    lv_keyfile_csv = r'H:\ML_LV\backup_keyfiles\keyfile_GE1956_QC.csv'
    dcm_folder = 'D:\DL_ECHO\LV_segmentation\predictions_dicom\Dec02_15-25-13_EFFIB0-DICBCE_AL_TF-CAMUSHM_ADAM_T-GE1956_HMLHML_V-NONE_TRANSFER-EP150+150_LR0.001_BS20_SCL1_OUT\original'

    keep_gls_count(dcm_folder, lv_keyfile_csv)