# ================================================================
# 0. Section: Imports
# ================================================================
import pickle

TEMP_FOLDER = 'tests/integration/fixtures/temp/'



# ================================================================
# 1. Section: Pickle Save/Load Utilities
# ================================================================
def save_object(obj, filename):
    """Saves a Python object to a file using pickle."""
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_object(filename):
    """Loads a Python object from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)