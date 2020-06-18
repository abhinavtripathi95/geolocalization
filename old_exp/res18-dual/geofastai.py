from fastai.vision import *
from fastai.callbacks import SaveModelCallback
import h5py

################################################################################################################
# sort the image path list
def atoi(text):
    if text.isdigit(): return int(text)
    else:              return text
    
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

def create_sorted_list(fpath):
    """ input = list of posix paths
    output = list of str paths sorted with natural keys
    """
    fpath = [str(path) for path in fpath]
    fpath.sort(key=natural_keys)
    return fpath

################################################################################################################
# generate matching and non-matching pairs
def get_pairs():
    with h5py.File('data_labels.h5', 'r') as f:
        print (list(f.keys()))
        match_array = np.asarray(list(f['match_array_40']))
    match_array = match_array[:2078,:2078]
    matching_pairs = np.transpose(np.nonzero(match_array))        
    # Generate same number of non-matching pairs
    non_match_pairs = []
    for i in range(len(match_array)):
        match_idx = np.nonzero(match_array[i,:])
        match_idx = np.asarray(match_idx)
        for j in range(match_idx.size):
            rand_idx = np.random.randint(len(match_array))
            while(rand_idx in match_idx):
                rand_idx = np.random.randint(len(match_array))
            non_match_pairs.append(np.array([i, rand_idx]))
    non_match_pairs = np.asarray(non_match_pairs)
    return matching_pairs, non_match_pairs


################################################################################################################
# class to get dataframe with uav and satellite images 
class AerialCity():
    def __init__(self, city):
        self.city = city
        self.sat_img_dir = 'train/' + self.city + '/sat-small'
        self.uav_img_dir = 'train/' + self.city + '/uav-small'
        
    def get_aerial_images(self):
        '''List of all the images, sorted by `natural_keys`'''
        fuav = get_image_files(self.uav_img_dir)
        fsat = get_image_files(self.sat_img_dir)
        fuav = create_sorted_list(fuav)
        fsat = create_sorted_list(fsat)
        return fuav, fsat

    def create_df(self):
        matching_pairs, non_match_pairs = get_pairs()
        fuav, fsat = self.get_aerial_images()
        uav_pathsdf = [fuav[idx] for idx in matching_pairs[:,0]] + [fuav[idx] for idx in non_match_pairs[:,0]]
        sat_pathsdf = [fsat[idx] for idx in matching_pairs[:,1]] + [fsat[idx] for idx in non_match_pairs[:,1]]
        n_matches = len(matching_pairs)
        assert n_matches == len(non_match_pairs)
        labeldf = [1] * n_matches + [0] * n_matches
        is_validdf = ([False] *(4*n_matches//5) + [True] *(n_matches - 4*n_matches//5))*2
        dicdf = {'uav'    : uav_pathsdf,
                'sat'     : sat_pathsdf,
                'label'   : labeldf,
                'is_valid': is_validdf}
        df = pd.DataFrame(dicdf)
        return df, n_matches

################################################################################################################
