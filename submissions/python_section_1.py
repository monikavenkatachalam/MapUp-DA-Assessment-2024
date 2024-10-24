from typing import Dict, List

import pandas as pd
import numpy as np
import polyline


#Question 1: Reverse List by N Elements

def reverse_by_n_elements(lst, n):
    result = []
    length = len(lst)

    for i in range(0, length, n):  
       chunk = lst[i:i + n]
       reversed_chunk = []
       for j in range(len(chunk)-1, -1, -1):
            reversed_chunk.append(chunk[j])
       result.extend(reversed_chunk)

    return result

# Example cases:
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))          
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  


#Question 2: Lists & Dictionaries

def group_strings_by_length(lst):
    result = {}

    for word in lst:
        length = len(word)

        
        if length not in result:
            result[length] = []

        result[length].append(word)

   
    sorted_result = dict(sorted(result.items()))

    return sorted_result

# Example cases:
print(group_strings_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
print(group_strings_by_length(["one", "two", "three", "four"]))


#Question 3: Flatten a Nested Dictionary

def flatten_dict(d, parent_key='', sep='.'):
    items = {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    items.update(flatten_dict(item, list_key, sep=sep))
                else:
                    items[list_key] = item
        else:
            items[new_key] = v

    return items

# Example input
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened_dict = flatten_dict(nested_dict)
for key,value in flattened_dict.items():
    print(f"{key}: {value}")

def unique_permutations(nums):
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        seen = set()  # To avoid picking duplicate elements in the same position
        for i in range(start, len(nums)):
            if nums[i] not in seen:
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]  # Swap
                backtrack(start + 1)
                nums[start], nums[i] = nums[i], nums[start]  # Swap back
    
    result = []
    nums.sort()  # Sort the input list to make sure duplicates are adjacent
    backtrack(0)
    return result

# Example case:
nums = [1, 1, 2]
perms = unique_permutations(nums)
for perm in perms:
    print(perm)

#Question 5: Find All Dates in a Text

import re

def find_all_dates(text):
    patterns = [
        r'\b(\d{2})-(\d{2})-(\d{4})\b',       
        r'\b(\d{2})/(\d{2})/(\d{4})\b',       
        r'\b(\d{4})\.(\d{2})\.(\d{2})\b'      
    ]
    combined_pattern = re.compile('|'.join(patterns))

    matches = combined_pattern.findall(text)
    valid_dates = []
    for match in matches:
        
        if match[0]:  
            valid_dates.append(f"{match[0]}-{match[1]}-{match[2]}")
        elif match[3]:  
            valid_dates.append(f"{match[3]}/{match[4]}/{match[5]}")
        elif match[6]: 
            valid_dates.append(f"{match[6]}.{match[7]}.{match[8]}")

    return valid_dates


text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
dates = find_all_dates(text)
print(dates)




def haversine(lat1, lon1, lat2, lon2):
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)*2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)*2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371000  
    return c * r

def decode_polyline_to_dataframe(polyline_str):
    decoded_coords = polyline.decode(polyline_str) 
  
    df = pd.DataFrame(decoded_coords, columns=['latitude', 'longitude'])
    
    distances = [0]  
    for i in range(1, len(df)):
        dist = haversine(df.latitude[i-1], df.longitude[i-1], df.latitude[i], df.longitude[i])
        distances.append(dist)
    
    df['distance'] = distances
    return df

polyline_str = "u{uFf`d@h@fD?ZP~u@~Z"

df = decode_polyline_to_dataframe(polyline_str)
print(df)


import numpy as np

def rotate_and_transform(matrix):
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Create a transformed matrix
    transformed_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate the sum of the row and column, excluding the current element
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            transformed_matrix[i][j] = row_sum + col_sum
    
    return transformed_matrix

# Example usage
input_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_transform(input_matrix)

# Print the output in a list of lists format
print(result)


import pandas as pd

# Step 1: Read the CSV file into a DataFrame
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("No data found in the CSV file.")
        return None
    except pd.errors.ParserError:
        print("Error parsing the CSV file.")
        return None

# Step 2: Inspect the DataFrame
def inspect_data(df):
    print("\nData Overview:")
    print(df.head())  # Print first few rows
    print("\nData Types:")
    print(df.dtypes)  # Print data types of columns
    print("\nMissing Values:")
    print(df.isnull().sum())  # Check for missing values

# Step 3: Check time completeness
def check_time_completeness(df):
    # Step 3a: Convert timestamp columns to datetime
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Step 3b: Group by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])

    # Initialize a list to hold the results
    results = []

    # Step 3c: Check each group for time coverage
    for (id_val, id_2_val), group in grouped:
        # Check if there is a full 24-hour coverage
        start_min = group['start_datetime'].min()
        end_max = group['end_datetime'].max()
        hours_covered = (end_max - start_min).total_seconds() / 3600

        # Check if all days of the week are covered
        days_covered = group['start_datetime'].dt.dayofweek.unique()
        full_week_coverage = len(days_covered) == 7

        # Check for full 24-hour coverage
        full_day_coverage = (hours_covered >= 24)

        # Add the result for this (id, id_2) pair
        results.append(((id_val, id_2_val), not (full_day_coverage and full_week_coverage)))

    # Step 3d: Create a boolean series with a multi-index
    boolean_series = pd.Series(dict(results)).unstack().stack().astype(bool)
    return boolean_series

# Example usage
file_path = 'dataset-1.csv'  # Replace with your actual CSV file path
df = load_data(file_path)

if df is not None:
    inspect_data(df)  # Optional: inspect the loaded data
    result = check_time_completeness(df)  # Process the data
    print(result)  # Output the result
