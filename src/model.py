import pickle

def save_model(model, filename="model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)