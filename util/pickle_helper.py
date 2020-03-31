try:
	import cPickle as pickle
except:
	import pickle

PICKLED_PATH = './data/pickled'

def get_pickled_data(filename, folder_path=PICKLED_PATH):
	path = f'{folder_path}/{filename}.pickle'
	with open(path, 'rb') as f:
		return pickle.load(f)

def store_data_as_pickle(data, filename, folder_path=PICKLED_PATH):
	path = f'{folder_path}/{filename}.pickle'
	with open(f'{path}', 'wb') as f:
			pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)