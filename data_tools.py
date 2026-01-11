import os
import pandas as pd

from data_analysis import calculate_lisas


def load_raw_trial_data(root_dir='.\\trial_data'):
    """
    Load all trial data
    :param root_dir: the path of "trial_data"
    :return:
    """
    raw_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.tsv'):
                parts = file.split('_')
                sub, task, run = parts[0], parts[2], parts[3]
                df_file = pd.read_csv(os.path.join(root, file), sep='\t')
                raw_list.append(
                    {
                        'subject': sub,
                        'task': task,
                        'run': run,
                        'lisas': calculate_lisas(df_file),
                        'acc': df_file['accuracy'].mean(),
                    }
                )

    df = pd.DataFrame(raw_list)
    return df

