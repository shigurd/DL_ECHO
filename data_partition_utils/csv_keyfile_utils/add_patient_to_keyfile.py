import os
import csv 
import argparse

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
    

def append_info_to_csv(input_csv, writer):
    
    gls_list = []
    
    with open(input_csv, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        header = next(csv_reader)
        
        for row in csv_reader:
            dicom_id, exam_id, projection, img_quality, mask_quality, data_setting = row            
            
            ''' this is for HCM patients exam_id_types that has date '''
            if len(exam_id.split('_')) == 3:
                patient_id = exam_id.split('_')[0]
            
            else:
                ''' this is for all other exam_id_types '''
                patient_id = exam_id
            
            ''' add to existing patient if projection found '''
            found = False
            for idx, x in enumerate(gls_list):    
                if x.get_name() == patient_id:
                    found = True
                    x.register_projection(projection)
                    patient_number = f'PATIENT{format(idx + 1, "04d")}'
                    patient_file = f'{format(sum(x.get_stats()), "02d")}'


            ''' add new patient if not found '''
            if found == False:
                temp = GLStracker(patient_id)
                temp.register_projection(projection)
                gls_list.append(temp)
                patient_number = f'PATIENT{format(len(gls_list), "04d")}'
                patient_file = f'{format(sum(temp.get_stats()), "02d")}'
            
            patient_id = f'{patient_number}_{patient_file}'
            writer.writerow([dicom_id, patient_id, exam_id, projection, img_quality, mask_quality, data_setting])
    
    checksum = 0
    gls_patients = 0

    for e in gls_list:
        if e.has_gls():
            gls_patients +=1
  
        print(e.get_name(), e.has_gls(), e.get_stats())
        checksum += sum(e.get_stats())
        
    print('total patients n =', len(gls_list)) 
    print('total patients with GLS n =', gls_patients)
    print('total exams', checksum)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('output_csv')
    args = parser.parse_args()
    
    output = open(args.output_csv, 'w', newline='')
    writer = csv.writer(output)
    writer.writerow(['dicom_id', 'patient_id', 'exam_id', 'projection', 'img_quality', 'mask_quality', 'data_setting'])
    append_info_to_csv(args.input_csv, writer)
    output.close()