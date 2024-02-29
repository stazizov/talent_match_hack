import json 
import tqdm 
import pandas as pd 
from tqdm import tqdm

def read_json(data):
    data = json.load(data)
    df = pd.json_normalize(data)

    return df


def transform_data(data):
    new_data = pd.DataFrame(columns=['uuid', 'birth_date', 'about', 'key_skills', 
                                     'employment_positions', 'employment_texts', 
                                     'faculty', 'organization', 'vacancy_name',
                                     'vacancy_description', 'label'])

    for i, row in tqdm(data.iterrows()):
        for j in range(len(row['failed_resumes'])):
            data_item = row['failed_resumes'][j]

            employment_positions = ''
            employment_texts = ''
            try:
                for m in range(len(data_item['experienceItem'])):
                    experience_item = data_item['experienceItem'][m]
                    employment_positions += f"{experience_item['position']} "
                    employment_texts += f"{experience_item['description']} "
            except:
                pass
            
            faculty = ''
            organization = ''
            try:
                for m in range(len(data_item['educationItem'])):
                    education_item = data_item['educationItem'][m]
                    faculty += f"{education_item['faculty']} "
                    organization += f"{education_item['organization']} "
            except:
                pass

            new_row = pd.Series({'uuid': data_item['uuid'], 'birth_date': data_item['birth_date'], 'about': data_item['about'],
                                 'key_skills': data_item['key_skills'], 'employment_positions': employment_positions, 
                                 'employment_texts': employment_texts, 'faculty': faculty, 'organization': organization,
                                 'vacancy_name': row['vacancy.name'], 'vacancy_description': row['vacancy.description'], 'label': 0})
            # new_data = new_data.append(new_row, ignore_index=True)
            new_data = pd.concat([new_data, pd.DataFrame([new_row])], ignore_index=True)

        for j in range(len(row['confirmed_resumes'])):
            data_item = row['confirmed_resumes'][j]

            employment_positions = ''
            employment_texts = ''
            try:
                for m in range(len(data_item['experienceItem'])):
                    experience_item = data_item['experienceItem'][m]
                    employment_positions += f"{experience_item['position']} "
                    employment_texts += f"{experience_item['description']} "
            except:
                pass

            faculty = ''
            organization = ''
            try:
                for m in range(len(data_item['educationItem'])):
                    education_item = data_item['educationItem'][m]
                    faculty += f"{education_item['faculty']} "
                    organization += f"{education_item['organization']} "
            except:
                pass

            new_row = pd.Series({'uuid': data_item['uuid'], 'birth_date': data_item['birth_date'], 'about': data_item['about'],
                                 'key_skills': data_item['key_skills'], 'employment_positions': employment_positions, 
                                 'employment_texts': employment_texts, 'faculty': faculty, 'organization': organization, 
                                 'vacancy_name': row['vacancy.name'], 'vacancy_description': row['vacancy.description'], 'label': 1})
            # new_data = new_data.append(new_row, ignore_index=True)
            new_data = pd.concat([new_data, pd.DataFrame([new_row])], ignore_index=True)

    return new_data