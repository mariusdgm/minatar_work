def merge_dictionaries(dict1, dict2):
    merged_dict = dict1.copy()  # Create a copy of the first dictionary
    
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] += value  # Add the values if the key exists
        else:
            merged_dict[key] = value  # Add the key-value pair if the key doesn't exist
    
    return merged_dict