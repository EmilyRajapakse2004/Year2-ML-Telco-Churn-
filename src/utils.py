import pickle

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
