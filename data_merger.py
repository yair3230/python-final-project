# Data merger.
import os
import json
import shutil

import pandas as pd

DATE_COLUMNS = [
    'T1w_date',
    'Syllogisms_run-01_date',
    'Syllogisms_run-02_date',
    'Transitive_run-01_date',
    'Transitive_run-02_date',
]


def convert_base_file():
    df = pd.read_csv('..\\ds002886\\participants.tsv', sep='\t')
    df['sex'] = df['sex'].replace(1, 'male')
    df['sex'] = df['sex'].replace(2, 'female')
    df['handedness'] = df['handedness'].replace(1, 'left')
    df['handedness'] = df['handedness'].replace(2, 'right')

    # Correct all dates
    for column_name in DATE_COLUMNS:
        df[column_name] = pd.to_datetime(df[column_name])
        df[column_name] = df[column_name] + pd.DateOffset(years=200)

    race_map = {
        "1": "American Indian or Alaskan Native",
        "2": "Asian",
        "3": "Black or African American",
        "4": "Native Hawaiian or Other Pacific Islander",
        "5": "White",
        "6": "Two or more races",
        "7": "Other",
    }
    for key, value in race_map.items():
        df['race'] = df['race'].replace(int(key), value)

    ethnicity_map = {"1": "Hispanic or Latino", "2": "Not Hispanic or Latino", "3": "Unknown or not resported"}
    for key, value in ethnicity_map.items():
        df['ethnicity'] = df['ethnicity'].replace(int(key), value)
    df.to_csv('main_dataset.csv')


def add_t1():
    main_df = pd.read_csv('main_dataset.csv')
    ses_t1_path = '..\\ds002886\\phenotype\\ses-T1'
    for filename in os.listdir(ses_t1_path):
        full_path = ses_t1_path + '\\' + filename
        df = pd.read_csv(full_path, delimiter='\t')
        main_df = pd.merge(main_df, df, left_on='participant_id', right_on='participant_id', how='left')
    main_df.to_csv('main_dataset_united.csv')


def add_t2():
    main_df = pd.read_csv('main_dataset.csv')
    ses_t1_path = '..\\ds002886\\phenotype\\ses-T2'
    for filename in os.listdir(ses_t1_path):
        full_path = ses_t1_path + '\\' + filename
        df = pd.read_csv(full_path, delimiter='\t')

        # Rename all columns except 'participant_id'
        df = df.rename(columns={col: f"{col}_t2" for col in df.columns if col != 'participant_id'})

        main_df = pd.merge(main_df, df, left_on='participant_id', right_on='participant_id', how='left')
    main_df.to_csv('main_dataset_united.csv')


def add_dtype_to_data_description():
    main_df = pd.read_csv('main_dataset.csv')
    with open('data_description.json') as openfile:
        desc = json.load(openfile)
    for column in main_df.columns:
        if column in DATE_COLUMNS:
            desc[column]['type'] = 'datetime64'
        elif column in desc:
            desc[column]['type'] = main_df[column].dtype.__str__()
    with open('data_description.json', 'w') as openfile:
        json.dump(desc, openfile)


def convert_columns_to_datetime():
    df = pd.read_csv('main_dataset.csv')
    for column_name in DATE_COLUMNS:
        df[column_name] = pd.to_datetime(df[column_name], dayfirst=True)
        print(df[column_name].dtype)
    # df.to_csv('main_dataset.csv', index=False)


def clean_empty_column():
    df = pd.read_csv('main_dataset.csv')
    df.dropna(axis=1, how='all', inplace=True)
    df.to_csv('main_dataset.csv')


def copy_session_data():
    data_path = '..\\ds002886'
    target_path = '.\\trial_data'
    os.mkdir(target_path)

    # Keep only the folders starting with 'sub-'
    folders = [x for x in os.listdir(data_path) if x.startswith('sub-')]

    for folder in folders:
        subject_target_folder = target_path + '\\' + folder
        subject_src_folder = data_path + '\\' + folder + '\\ses-T1\\func'
        os.mkdir(subject_target_folder)

        # Get the names of all data files in the src folder
        data_files = [file for file in os.listdir(subject_src_folder) if file.endswith('.tsv')]
        for file in data_files:
            src_file = subject_src_folder + '\\' + file
            dst_file = subject_target_folder + '\\' + file
            shutil.copy2(src_file, dst_file)


# convert_base_file()
# add_t1()
# add_t2()
# add_dtype_to_data_description()
# convert_columns_to_datetime()
# clean_empty_column()
# copy_session_data()
