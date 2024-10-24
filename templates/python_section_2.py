import pandas as pd
import datetime as dt


#Que 9:

import pandas as pd

def unroll_distance_matrix(df):
    
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))

    
    all_combinations = pd.MultiIndex.from_product([unique_ids, unique_ids], names=["id_start", "id_end"]).to_frame(index=False)

    
    result = pd.merge(all_combinations, df, how='left', on=['id_start', 'id_end'])

    
    result = result[result['id_start'] != result['id_end']]

    return result


try:
    df = pd.read_csv("D/datasetsataset-2")  
    unrolled_df = unroll_distance_matrix(df)
    print(unrolled_df)
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")

#que:10

def calculate_distance_matrix(df):
def unroll_distance_matrix(df: pd.DataFrame):
   
    unrolled_data = []
    
    
    ids = df['id'].values  
    
    
    for i, id_start in enumerate(ids):
        for j, id_end in enumerate(ids):
            if id_start != id_end:  
                distance = df.iloc[i, j + 1]  
                unrolled_data.append([id_start, id_end, distance])
    
    
    result_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    
    return result_df

print(result)


#que:11


def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_value: int):
   
    reference_avg = df[df['id_start'] == reference_value]['id_start'].mean()

    
    lower_bound = reference_avg * 0.9
    upper_bound = reference_avg * 1.1

    
    ids_within_threshold = df[(df['id_start'] >= lower_bound) & (df['id_start'] <= upper_bound)]['id_start']

   
    return sorted(ids_within_threshold)



que:12

def calculate_toll_rate(df: pd.DataFrame):
    
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    
    df['moto'] = df['distance'] * rates['moto']
    df['car'] = df['distance'] * rates['car']
    df['rv'] = df['distance'] * rates['rv']
    df['bus'] = df['distance'] * rates['bus']
    df['truck'] = df['distance'] * rates['truck']
    
    return df


que:13

def calculate_time_based_toll_rates(df_ngu: pd.DataFrame):
    
    start_days = []
    end_days = []
    start_times = []
    end_times = []
    modified_rates = []

    
    weekday_discounts = {
        '00:00:00-10:00:00': 0.8,
        '10:00:00-18:00:00': 1.2,
        '18:00:00-23:59:59': 0.8
    }
    weekend_discount = 0.7  

    
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']
    
   
    time_intervals = [
        (dt.time(0, 0), dt.time(10, 0)),
        (dt.time(10, 0), dt.time(18, 0)),
        (dt.time(18, 0), dt.time(23, 59))
    ]
    
    
    for index, row in df_ngu.iterrows():
        base_rate = row['toll_rate'] 
        
        
        for day in weekdays + weekends:
            for (start_time, end_time) in time_intervals:
                start_days.append(day)
                end_days.append(day)
                start_times.append(start_time)
                end_times.append(end_time)
                
                
                if day in weekdays:
                    if start_time == dt.time(0, 0):
                        discount_factor = weekday_discounts['00:00:00-10:00:00']
                    elif start_time == dt.time(10, 0):
                        discount_factor = weekday_discounts['10:00:00-18:00:00']
                    else:
                        discount_factor = weekday_discounts['18:00:00-23:59:59']
                else:
                    discount_factor = weekend_discount
                
               
                modified_rate = base_rate * discount_factor
                modified_rates.append(modified_rate)
    
    
    result_df = pd.DataFrame({
        'start_day': start_days,
        'end_day': end_days,
        'start_time': start_times,
        'end_time': end_times,
        'modified_toll_rate': modified_rates
    })
    
    
    final_df = pd.concat([df_ngu, result_df], axis=1)
    
    return final_df

