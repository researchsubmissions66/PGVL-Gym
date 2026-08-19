from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import Dataset
import h5py

from common.utils.utils import generate_split, nth
from common.configuration import expand_path


def _load_feature_tensor(path, feature_key='features'):
	"""Load a patch bag from either the native HDF5 store or a torch file."""
	path = os.path.expanduser(expand_path(path))
	if path.lower().endswith(('.h5', '.hdf5')):
		with h5py.File(path, 'r') as hdf5_file:
			keys = [feature_key, 'features', 'embeddings', 'feats']
			key = next((key for key in keys if key and key in hdf5_file), None)
			if key is None:
				raise ValueError('No feature tensor found in {}'.format(path))
			features = torch.from_numpy(hdf5_file[key][:])
	else:
		try:
			features = torch.load(path, map_location='cpu', weights_only=True)
		except TypeError:  # torch < 2.0
			features = torch.load(path, map_location='cpu')
		if isinstance(features, dict):
			keys = [feature_key, 'features', 'embeddings', 'feats']
			key = next((key for key in keys if key and key in features), None)
			if key is None:
				raise ValueError('No feature tensor found in {}'.format(path))
			features = features[key]
	if not torch.is_tensor(features):
		raise TypeError('Unsupported feature payload in {}'.format(path))
	features = features.detach().float()
	if features.ndim == 3 and features.shape[0] == 1:
		features = features.squeeze(0)
	if features.ndim != 2 or features.shape[0] == 0:
		raise ValueError(
			'Expected a non-empty [patches, features] tensor in {}, got {}'.format(
				path, tuple(features.shape)))
	if not torch.isfinite(features).all():
		raise ValueError('Feature tensor contains NaN or infinity: {}'.format(path))
	return features

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
				 csv_path = 'dataset_csv/ccrcc_clean.csv',
				 mode = 'clam',
				 shuffle = False,
				 seed = 7,
				 print_info = True,
				 label_dict = None,
				 filter_dict = None,
				 ignore=None,
				 patient_strat=False,
				 label_col = None,
				 patient_voting = 'max',
				 ):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		label_dict = {} if label_dict is None else label_dict
		filter_dict = {} if filter_dict is None else filter_dict
		ignore = [] if ignore is None else ignore
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir_s = None
		self.data_dir_l = None
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path)
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		if shuffle:
			slide_data = slide_data.sample(
				frac=1.0, random_state=seed).reset_index(drop=True)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.mode = mode
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def cls_ids_prep(self):
		self.patient_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) 
		patient_labels = []

		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() 
			elif patient_voting == 'maj':
				label = stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)

		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data

	def filter_df(self, df, filter_dict=None):
		filter_dict = {} if filter_dict is None else filter_dict
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
			'n_splits' : k,
			'val_num' : val_num,
			'test_num': test_num,
			'label_frac': label_frac,
			'seed': self.seed,
			'custom_test_ids': custom_test_ids
		}

		if self.patient_strat:
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))]

			for split in range(len(ids)):
				for idx in ids[split]:
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			requested = split.astype(str).tolist()
			seen = set()
			duplicates = set()
			for value in requested:
				if value in seen:
					duplicates.add(value)
				seen.add(value)
			duplicates = sorted(duplicates)
			if duplicates:
				raise ValueError(
					'{} split repeats slide IDs: {}'.format(
						split_key, ', '.join(duplicates[:3])))
			available = self.slide_data['slide_id'].astype(str)
			duplicate_annotations = sorted(
				set(available[available.duplicated()].tolist()))
			if duplicate_annotations:
				raise ValueError(
					'dataset manifest repeats slide IDs: {}'.format(
						', '.join(duplicate_annotations[:3])))
			missing = sorted(set(requested) - set(available.tolist()))
			if missing:
				raise ValueError(
					'{} split contains slide IDs absent from the dataset '
					'manifest: {}'.format(split_key, ', '.join(missing[:3])))
			mask = available.isin(requested)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode, num_classes=self.num_classes, feature_path_column_s=getattr(self, 'feature_path_column_s', None), feature_path_column_l=getattr(self, 'feature_path_column_l', None), feature_key=getattr(self, 'feature_key', 'features'), include_metadata=getattr(self, 'include_metadata', False))
		else:
			split = None

		print(len(split) if split is not None else 0)

		return split

	def get_merged_split_from_df(self, all_splits, split_keys=None):
		split_keys = ['train'] if split_keys is None else split_keys
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(merged_split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode, num_classes=self.num_classes, feature_path_column_s=getattr(self, 'feature_path_column_s', None), feature_path_column_l=getattr(self, 'feature_path_column_l', None), feature_key=getattr(self, 'feature_key', 'features'), include_metadata=getattr(self, 'include_metadata', False))
		else:
			split = None

		return split

	def return_splits(self, from_id=True, csv_path=None):

		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode, num_classes=self.num_classes, feature_path_column_s=getattr(self, 'feature_path_column_s', None), feature_path_column_l=getattr(self, 'feature_path_column_l', None), feature_key=getattr(self, 'feature_key', 'features'), include_metadata=getattr(self, 'include_metadata', False))

			else:
				train_split = None

			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode, num_classes=self.num_classes, feature_path_column_s=getattr(self, 'feature_path_column_s', None), feature_path_column_l=getattr(self, 'feature_path_column_l', None), feature_key=getattr(self, 'feature_key', 'features'), include_metadata=getattr(self, 'include_metadata', False))

			else:
				val_split = None

			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode, num_classes=self.num_classes, feature_path_column_s=getattr(self, 'feature_path_column_s', None), feature_path_column_l=getattr(self, 'feature_path_column_l', None), feature_key=getattr(self, 'feature_key', 'features'), include_metadata=getattr(self, 'include_metadata', False))

			else:
				test_split = None


		else:
			assert csv_path
			# all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  
			all_splits = pd.read_csv(csv_path, dtype={'dir': str, 'case_id': str, 'slide_id': str, 'label': str})
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test') 

		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							  columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]

		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		# assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1)
		df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
				 data_dir_s=None,
				 data_dir_l=None,
				 mode='clam',
				 feature_path_column_s=None,
				 feature_path_column_l=None,
				 feature_key='features',
				 include_metadata=False,
				 **kwargs):

		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir_s = data_dir_s
		self.data_dir_l = data_dir_l
		self.mode = mode
		self.feature_path_column_s = feature_path_column_s
		self.feature_path_column_l = feature_path_column_l
		self.feature_key = feature_key
		self.include_metadata = include_metadata
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data['label'][idx]
		if isinstance(self.data_dir_s, dict) and isinstance(self.data_dir_l, dict):
			source = self.slide_data['source'][idx]
			data_dir_s = self.data_dir_s[source]
			data_dir_l = self.data_dir_l[source]
		else:
			data_dir_s = self.data_dir_s
			data_dir_l = self.data_dir_l

		if not self.use_h5:
			if self.mode != 'transformer':
				return slide_id, label
			if self.feature_path_column_s and self.feature_path_column_l:
				path_s = self.slide_data[self.feature_path_column_s].iloc[idx]
				path_l = self.slide_data[self.feature_path_column_l].iloc[idx]
			elif data_dir_s and data_dir_l:
				path_s = os.path.join(data_dir_s, '{}.pt'.format(slide_id))
				path_l = os.path.join(data_dir_l, '{}.pt'.format(slide_id))
			else:
				raise ValueError(
					'Dual-scale data requires both feature path columns or both feature directories')
			features_s = _load_feature_tensor(path_s, self.feature_key)
			features_l = _load_feature_tensor(path_l, self.feature_key)
			if self.include_metadata:
				metadata = {
					'slide_id': str(slide_id),
					'case_id': str(self.slide_data['case_id'].iloc[idx]),
				}
				return features_s, features_l, metadata, label
			return features_s, features_l, label

		else:
			full_path = os.path.join(data_dir_s, 'h5_files', '{}.h5'.format(slide_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			return features, label, coords


class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, data_dir_s=None, data_dir_l=None, mode='clam', num_classes=2, feature_path_column_s=None, feature_path_column_l=None, feature_key='features', include_metadata=False):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir_s = data_dir_s
		self.data_dir_l = data_dir_l
		self.mode = mode
		self.feature_path_column_s = feature_path_column_s
		self.feature_path_column_l = feature_path_column_l
		self.feature_key = feature_key
		self.include_metadata = include_metadata
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)
