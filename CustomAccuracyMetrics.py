from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# a collection of accuracy metrics for our temporal data

# Definitions
# - Window: slice of sequential fetch targets used as input to the model
# - Mispred window: sequence of fetch targets between two sequential resteers

# Options for eval

# Calculating raw penalty
# - use penalty from first incorrect prediction in a window
# - use every incorrect prediction in a window

# normalizing raw penalty
# - normalize by number of resteer windows
# - normalize by number of fetch targets
# - normalize by number of correct resteer windows + incorrect predictions
# - normalize by number of on-path fetch targets
# - normalize by number of off-path fetch targets

# same as confidence mechanism implemented in scarab
# normalized by number of off-path cycles

def get_cycle_penalty_max_offpath(df, workload, max_penalty):
    groups_with_ones = df[df['prediction'] == 1].groupby('window_id').first().reset_index()
    groups_without_ones = df[~df['window_id'].isin(groups_with_ones['window_id'])]
    last_rows_no_ones = groups_without_ones.groupby('window_id').last().reset_index()

    # Combine the two results
    mispred_penalty_rows = pd.concat([groups_with_ones, last_rows_no_ones]).sort_values(by='window_id').reset_index(drop=True)
    earlies = mispred_penalty_rows[mispred_penalty_rows['label'] == 0]
    lates = mispred_penalty_rows[mispred_penalty_rows['label'] == 1]

    early_cycles = (earlies['cycle_penalty'].apply(lambda x: min(x, max_penalty))).sum()
    late_cycles = (lates['cycle_penalty'].apply(lambda x: min(x, max_penalty))).sum()
    # total penalty, early, late, off-path cycles, n mispreds
    return (mispred_penalty_rows['cycle_penalty'].apply(lambda x: min(x, max_penalty))).sum(), early_cycles, late_cycles, mispred_penalty_rows['off_path_len_cycles'].sum(), len(mispred_penalty_rows)

def get_cycle_penalty(df, workload):
    # Find groups with at least one 'Value' == 1
    groups_with_ones = df[df['prediction'] == 1].groupby('window_id').first().reset_index()
    # Find the last row for groups with no 'Value' == 1
    groups_without_ones = df[~df['window_id'].isin(groups_with_ones['window_id'])]
    last_rows_no_ones = groups_without_ones.groupby('window_id').last().reset_index()

    # Combine the two results
    mispred_penalty_rows = pd.concat([groups_with_ones, last_rows_no_ones]).sort_values(by='window_id').reset_index(drop=True)
    return mispred_penalty_rows['cycle_penalty'].sum(), mispred_penalty_rows['off_path_len_cycles'].sum(), len(mispred_penalty_rows)

N_SKIP = 0
def normalized_first_wrong(predictions, labels, penalties, off_path_lens, window_ids):
    prev_off_path_length = off_path_lens[0]
    total_off_path_length = 0
    total_penalty = 0
    cur_window_id = 0
    off_path = False
    prev_label = 1
    n_skipped = 0
    for pred, label, penalty, off_path_len, window_id in zip(predictions, labels, penalties, off_path_lens, window_ids):
        if window_id != cur_window_id:
            off_path = False
            cur_window_id = window_id
            total_off_path_length += prev_off_path_length
            prev_off_path_length = off_path_len
            n_skipped = 0
        if n_skipped < N_SKIP:
            n_skipped += 1
            continue
        if(off_path):
            continue
        if(label == 0):
            if pred == 1:
                off_path = True
                total_penalty += penalty
                continue
        if(label == 1):
            if not off_path:
                if pred == 0:
                    total_penalty += 1
                elif pred == 1:
                    off_path = True
    tot_fts = len(predictions)
    # penalty, tot fts, on_path_fts, off_path_fts
    # off_path_len sus!, doen't get last window??
    return total_penalty, tot_fts, tot_fts - total_off_path_length, total_off_path_length, len(np.unique(np.array(window_ids)))

# this accuracy doesn't make sense so I'll only look at this if I need to 
# measures each penatly independently, as if each happened in its own window
def raw_penalty_all_errors(predictions, labels, penalties, window_ids, off_path_lens):
    for workload in predictions.keys():
        # copy stuff out by workload
        predictions_w = predictions[workload]
        labels_w = labels[workload]
        penalties_w = penalties[workload]
        window_ids_w = window_ids[workload]
        off_path_lens_w = off_path_lens[workload]
        # 
        total_penalty = 0
        window_error_counts = defaultdict(int)
        for pred, label, penalty, window_id, off_path_len in zip(predictions_w, labels_w, penalties_w, window_ids_w, off_path_lens_w):
            if(pred != label):
                total_penalty += penalty
                window_error_counts[window_id] += 1
            print(f"{workload}: {total_penalty}")

        n_windows_wo_error = 0
        n_errors = 0
        for window in window_ids[workload]:
            if window_error_counts[window] == 0:
                n_windows_wo_error += 1
            else:
                n_errors += window_error_counts[window]
        print(f"{workload}: {total_penalty}")


# for testing
if __name__ == "__main__":
    df = pd.read_csv('clang.csv').head(1000)
    predictions = df['prediction'].to_list()
    labels = df['label'].to_list()
    penalties = df['penalty'].to_list()
    off_path_lens = df['off_path_len'].to_list()
    window_ids = df['window_id'].to_list()


    print(normalized_first_wrong(predictions, labels, penalties, off_path_lens, window_ids))