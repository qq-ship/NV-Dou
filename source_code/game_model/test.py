import pickle

path = 'model/mm0.64/w0nework01.pkl'

f = open(path, 'rb')

data = pickle.load(f)

print(data)
