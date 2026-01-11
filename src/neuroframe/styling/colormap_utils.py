import numpy as np

def get_id_division(id: int, searching_array: np.ndarray, segment_hierarchy: np.ndarray):
    # it is present
    if(np.isin(id, searching_array)): return np.where(searching_array == id)[0][0]

    else:
        next_pos = np.where(segment_hierarchy == id)[0][0] + 1

        if(next_pos >= len(segment_hierarchy)): return np.nan
        
        next_id = segment_hierarchy[next_pos] 
        return get_id_division(next_id, searching_array, segment_hierarchy)
    
def get_separators(separator_reference: np.ndarray, segments_present: np.ndarray, full_hierarchy: np.ndarray):
    separators = []
    for child in separator_reference: separators.append(get_id_division(child, segments_present, full_hierarchy))

    separators = np.array(separators)
    
    return separators
