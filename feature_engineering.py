import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
# Convert to a DataFrame
#df = pd.DataFrame(load_drawing_data(metadata, drawings_path))

def select_features(df, target_col='label', alpha=0.1):
  
    # Drop or summarize complex fields if necessary
    #df_without_drawings = df.drop(columns=['drawings'])

    #Defining new columns that represent additional features
    df['stroke duration'] = np.nan #refers to the time needed for each participant to complete the task, calculated as the difference between start time and end ti,e
    df["NCV"] = np.nan #number of changes in direction of velocity, calculated by adding up the number of local extrema
    df["V_max"] = np.nan
    df["V_min"] = np.nan
    df["p_min"] = np.nan
    df["p_max"] = np.nan
    df["NCP"] = np.nan #number of changes in pressure, calculated by adding up the number of local extrema
    df["V_std"] = np.nan
    df["V_mean"] = np.nan
    df["p_std"] = np.nan
    df["p_mean"] = np.nan

    df=df[(df['UPDRS V']<=3) | (df['UPDRS V'].isna()==True)]
    df = df.reset_index(drop=True)

    df['V_max'] = df['V_max'].astype(object)
    df['V_min'] = df['V_min'].astype(object)
    df['p_max'] = df['p_max'].astype(object)
    df['p_min'] = df['p_min'].astype(object)
    df['NCV'] = df['NCV'].astype(object)
    df['NCP'] = df['NCP'].astype(object)
    df['stroke duration'] = df['stroke duration'].astype(object)
    df["V_std"] = df["V_std"].astype(object)

    df["V_mean"] = df["V_mean"].astype(object)
    df["p_std"] = df["p_std"].astype(object)
    df["p_mean"] = df["p_mean"].astype(object)
    df["NCP"] = df["NCP"].astype(object)

    original_drawings = copy.deepcopy(df['drawings'])
    # Reset 'drawings' in DataFrame to original
    df['drawings'] = original_drawings

    # Assuming df and original_drawings are already defined
    for i in range(len(df)):
      
        taskovi = []
        drawings_list = original_drawings.iloc[i]  # Extract the drawings list for the current ID
        stroke_duration_list = []
        v_max_list = []
        v_min_list = []
        p_max_list = []
        p_min_list = []
        ncv_list = []
        ncp_list = []
        stroke_duration_list = []
        v_mean_list = []
        v_std_list = []
        p_mean_list = []
        p_std_list = []
        
        for j in range(len(drawings_list)):  # Iterate over each task
            task = drawings_list[j][0]  # Extract task data
            updated_task = []
            start_time = task[0][2]
            end_time = task[-1][2]
            stroke_duration_list.append((end_time - start_time) / 1000)

            v_res_list = []
            p_list = []

            # Collect pressure values to calculate a dynamic threshold
            for k in range(len(task) - 1):
                p_list.append(task[k][6])  # Pressure values for dynamic threshold

            # Calculate the 10th percentile as the dynamic threshold
            low_threshold = np.percentile(p_list, 10) if len(p_list) > 0 else 0

            # Iterate again to calculate velocities and apply the dynamic threshold
            for k in range(len(task) - 1):
                y_1, x_1, t_1 = task[k][0], task[k][1], task[k][2]
                y_2, x_2, t_2 = task[k + 1][0], task[k + 1][1], task[k + 1][2]

                v_y = (y_2 - y_1) / (t_2 - t_1)
                v_x = (x_2 - x_1) / (t_2 - t_1)
                v_res = math.sqrt(pow(v_y, 2) + pow(v_x, 2))

                v_res_list.append(v_res)

                # Apply the dynamic threshold for pressure
                if task[k][6] > low_threshold:
                    updated_row = np.append(task[k], [v_y, v_x, v_res])
                    updated_task.append(updated_row)

            # Append descriptive statistics
            v_max_list.append(max(v_res_list, default=np.nan))
            v_min_list.append(min(v_res_list, default=np.nan))
            p_max_list.append(max(p_list, default=np.nan))
            p_min_list.append(min(p_list, default=np.nan))
            v_mean_list.append(np.mean(v_res_list) if v_res_list else np.nan)
            v_std_list.append(statistics.stdev(v_res_list) if len(v_res_list) > 1 else np.nan)
            p_mean_list.append(np.mean(p_list) if p_list else np.nan)
            p_std_list.append(statistics.stdev(p_list) if len(p_list) > 1 else np.nan)

            peaks, _ = signal.find_peaks(np.array(v_res_list), width=10, distance=10)
            valleys, _ = signal.find_peaks(-np.array(v_res_list), width=10, distance=10)
            peaks_p, _ = signal.find_peaks(np.array(p_list), width=10, distance=10)
            valleys_p, _ = signal.find_peaks(-np.array(p_list), width=10, distance=10)

            ncv_list.append(len(peaks) + len(valleys))
            ncp_list.append(len(peaks_p) + len(valleys_p))

            # Add the last row without modifications
            updated_task.append(np.append(task[-1], [np.nan, np.nan, np.nan]))
            taskovi.append(updated_task)

        # Update DataFrame with calculated values
        df.at[i, 'drawings'] = taskovi
        df.at[i, 'V_max'] = v_max_list
        df.at[i, 'V_min'] = v_min_list
        df.at[i, 'p_max'] = p_max_list
        df.at[i, 'p_min'] = p_min_list
        df.at[i, 'stroke duration'] = stroke_duration_list
        df.at[i, 'NCV'] = ncv_list
        df.at[i, 'NCP'] = ncp_list
        df.at[i, "V_mean"] = v_mean_list
        df.at[i, "V_std"] = v_std_list
        df.at[i, "p_std"] = p_std_list
        df.at[i, "p_mean"] = p_mean_list
        
    features = ['V_max', 'p_max', 'stroke duration', 'NCV', 'NCP', "V_mean", "V_std", "p_std", "p_mean"]
    df_features = df[features]

    # Filter healthy and Parkinson datasets
    df_healthy = df[df['Disease']=="H"]
    df_healthy = df_healthy.reset_index()
    df_parkinson = df[df['Disease']=="PD"]
    df_parkinson = df_parkinson.reset_index()


    healthy_bootstrap = resample(df_healthy, n_samples=33, random_state=42)
    parkinson_bootstrap = resample(df_parkinson, n_samples=33, random_state=42)

    healthy_bootstrap = healthy_bootstrap.reset_index()
    parkinson_bootstrap = parkinson_bootstrap.reset_index()

    # Initialize parameters
    #alpha = 0.1
    dict_features = {}

    # Iterate over features and tasks
    for column in df_features:
        for j in range(8):  # Assuming 8 tasks
            feature_list_healthy = []
            feature_list_pa = []

            # Collect feature values for each task
            for i in range(int(len(healthy_bootstrap[['ID', 'drawings']]))):
                feature_list_healthy.append(healthy_bootstrap[column][i][j])
                feature_list_pa.append(parkinson_bootstrap[column][i][j])

            # Skip if one of the lists contains identical values
            if len(set(feature_list_healthy)) == 1 or len(set(feature_list_pa)) == 1:
                print(f"Identical values found in one of the lists! Task {j}, Column {column}")
                continue

            # Perform Mann-Whitney U test
            U1, p = mannwhitneyu(feature_list_healthy, feature_list_pa, method="exact")

            if p < alpha:
                # Store significant feature
                dict_features[f"{column} task_{j + 1}"] = p
                
                # Plot boxplot
                data_to_plot = pd.DataFrame({
                    'Values': feature_list_healthy + feature_list_pa,
                    'Group': ['Healthy'] * len(feature_list_healthy) + ['Parkinson'] * len(feature_list_pa)
                })

                plt.figure(figsize=(4, 3))
                sns.boxplot(x="Group", y="Values", data=data_to_plot, palette="Set2")
        
                    
                plt.xlabel("Group")
                plt.ylabel(f"{column} Task {j + 1}")
                plt.show()

    # Convert significant features to a DataFrame
    df_sig = pd.DataFrame(dict_features.items(), columns=['Feature', 'p-value'])

    return df_sig
