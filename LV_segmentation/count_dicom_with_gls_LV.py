import os
import csv
from data_partition_utils.random_testsample_kfold import split_text_on_first_number

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


class StudyPatientTracker:
    def __init__(self, study_tag):
        self.name = study_tag
        self.study_list = []
        self.patient_list = []

    def append_patient(self, new_patient):
        self.patient_list.append(new_patient)

    def append_study(self, new_study):
        self.study_list.append(new_study)

    def get_study_tag(self):
        return self.name

    def get_study_list(self):
        return self.study_list

    def get_patient_list(self):
        return self.patient_list


def count_patients_in_study(csv_file):
    studies_found = []

    with open(csv_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        header = next(csv_reader)

        for row in csv_reader:
            file_id, patient_id, exam_id, projection, img_quality, mask_quality, data_setting = row

            for i, c in enumerate(exam_id):
                if c in [str(number) for number in range(9)]:
                    split_idx = i
                    break

            study_tag = exam_id[:split_idx]
            patient_without_img_id = patient_id.rsplit('_', 1)[0]
            exam_without_img_id = exam_id.split('_', 1)[0]

            in_studies = False
            for study in studies_found:
                if study_tag == study.get_study_tag():
                    in_studies = True
                    if patient_without_img_id not in study.get_patient_list():
                        study.append_patient(patient_without_img_id)
                    if exam_without_img_id not in study.get_study_list():
                        study.append_study(exam_without_img_id)

            if in_studies == False:
                new_study = StudyPatientTracker(study_tag)
                new_study.append_patient(patient_without_img_id)
                new_study.append_study(exam_without_img_id)
                studies_found.append(new_study)

    for s in studies_found:
        print(s.get_study_tag(), 'n patients =', len(s.get_patient_list()), 'n study =', len(s.get_study_list()))


def csv_input_gls_count(csv_file):
    gls_list = []
    count_hcm_patients = []
    count_angio_patients = []

    with open(csv_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        header = next(csv_reader)

        for row in csv_reader:
            file_id, patient_id, exam_id, projection, img_quality, mask_quality, data_setting = row

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


def get_train_val_test_info(img_folder_pth, keyfile_csv):
    gls_list = []

    for fn in os.listdir(img_folder_pth):
        patient_id_folder = fn.rsplit('_', 4)[0]

        with open(keyfile_csv, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            header = next(csv_reader)

            found_in_csv = False
            for row in csv_reader:
                file_id, patient_id, exam_id, projection, img_quality, mask_quality, data_setting = row
                if patient_id_folder == patient_id:
                    found_in_csv = True

                    found = False
                    for x in gls_list:
                        if x.get_name() == exam_id:
                            found = True
                            x.register_projection(projection)

                    if found == False:
                        temp = GLStracker(exam_id)
                        temp.register_projection(projection)
                        gls_list.append(temp)

            if found_in_csv == False:
                print(fn, 'not found in csv')


    studies = dict()
    for gls in gls_list:
        study_group = split_text_on_first_number(gls.get_name())[0]

        if study_group not in studies:
            studies[study_group] = [gls]
        else:
            studies[study_group].append(gls)

    [[print(j.get_name()) for j in studies[i]] for i in studies]
    for key in studies:
        dict_list = studies[key]
        print(f'{key} exams = {len(studies[key])}')
        gls_in_study = 0
        for item in dict_list:
            if item.has_gls() == True:
                gls_in_study += 1
                #print(item.get_name())
        print(f'{key} exams that has gls = {gls_in_study}')


if __name__ == "__main__":
    lv_keyfile_csv = r'H:\ML_LV\backup_keyfiles\keyfile_GE1965_QC_CRIDP.csv'
    dcm_folder = 'D:\DL_ECHO\LV_segmentation\predictions_dicom\Dec02_15-25-13_EFFIB0-DICBCE_AL_TF-CAMUSHM_ADAM_T-GE1956_HMLHML_V-NONE_TRANSFER-EP150+150_LR0.001_BS20_SCL1_OUT\original'

    img_folder = r'D:\DL_ECHO\LV_segmentation\data\test\imgs\GE1956_HMLHML'

    get_train_val_test_info(img_folder, lv_keyfile_csv)
    #keep_gls_count(dcm_folder, lv_keyfile_csv)
    #count_patients_in_study(lv_keyfile_csv)
