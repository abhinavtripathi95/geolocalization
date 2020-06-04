from fastai.vision import *
from fastai.callbacks import SaveModelCallback
import h5py
import torchvision

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
        self.sat_img_dir = 'train/' + self.city + '/sat300'
        self.uav_img_dir = 'train/' + self.city + '/uav' 

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
# normalization helper functions
def get_stats(cam):
    uav_m, uav_s = (torch.tensor([0.4719, 0.4783, 0.4521]), 
                    torch.tensor([0.2227, 0.2020, 0.2070]))
    sat_m, sat_s = (torch.tensor([0.3796, 0.3894, 0.3490]), 
                    torch.tensor([0.2137, 0.2030, 0.2064]))
    if   cam == 'uav': return (uav_m, uav_s)
    elif cam == 'sat': return (sat_m, sat_s)
    
################################################################################################################
# dataset class
class GeoDataset(Dataset):
    """Dataset of satellite and uav images."""
    path = '.'
    device = 'cuda'
    def __init__(self, df, uav_tfm=None, sat_tfm=None, xtra_tfm=None):
        """
        Args:
            df (dataframe): Path of files with labels and validation set annotation.
            uav_tfm, sat_tfm (callable, optional): Optional transform to be applied
                on a uav or satellite sample.
             xtra_tfm (callable, optional): transform on all items, use this to convert
                 to torch tensor
        """
        self.df = df
        self.uav_tfm = uav_tfm
        self.sat_tfm = sat_tfm
        self.xtra_tfm = xtra_tfm
        self.c = 1. # number of outputs
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        puav, psat, label = (self.df.iloc[idx, 0],
                             self.df.iloc[idx, 1],
                             self.df.iloc[idx, 2])
        # load both images and apply tfms
        imuav, imsat = PIL.Image.open(puav), PIL.Image.open(psat)
#         sample = {'imuav': imuav, 'imsat': imsat, 'label':label}
        sample = {'imgs': [imuav,imsat], 'label':label}
        if self.uav_tfm: sample['imgs'][0] = self.uav_tfm(sample['imgs'][0])
        if self.sat_tfm: sample['imgs'][1] = self.sat_tfm(sample['imgs'][1])
        if self.xtra_tfm: sample = self.xtra_tfm(sample)
        return sample

################################################################################################################
# all the transforms used for the images
class Rescale(object):
    """Rescale the image to sz*sz"""
    def __init__(self, sz):
        assert isinstance(sz, int)
        self.sz = sz
    def __call__(self, img):
        new_h, new_w = self.sz, self.sz
        new_img = img.resize((new_h, new_w))
        return new_img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        imuav, imsat, label = (np.asarray(sample['imgs'][0]),
                               np.asarray(sample['imgs'][1]), 
                               np.asarray(sample['label']))

        uav_m, uav_s = get_stats('uav')
        uav_m, uav_s = uav_m[...,None,None], uav_s[...,None,None]
        sat_m, sat_s = get_stats('sat')
        sat_m, sat_s = sat_m[...,None,None], sat_s[...,None,None]

        # swap color axis: numpy HWC to torch CHW
        imuav = np.transpose(imuav, (2, 0, 1)) # note that range is 0-255 not 0-1
        imsat = np.transpose(imsat, (2, 0, 1))
        return [ [(torch.from_numpy(imuav).float()/255 - uav_m)/uav_s,
                  (torch.from_numpy(imsat).float()/255 - sat_m)/sat_s],
                torch.from_numpy(label).float() ]

################################################################################################################
# make dataset
def get_ds(df):
    # torchvision transforms can be used when a composition is required
    train_ds = GeoDataset(df[df['is_valid']==False],
                        uav_tfm=Rescale(224),
                        sat_tfm=Rescale(224),
                        xtra_tfm=ToTensor())
    #                          transform = torchvision.transforms.Compose([Rescale(256),
    #                                                         ToTensor()
    #                                                         ]))
    valid_ds = GeoDataset(df[df['is_valid']==True],
                        uav_tfm=Rescale(224),
                        sat_tfm=Rescale(224),
                        xtra_tfm=ToTensor())
    return train_ds, valid_ds

################################################################################################################
# functions to view the contents of the dataset
def deprocess(imgt, cam):
    """Convert from torch tensor to numpy for display"""
    mean, std = get_stats(cam)
    mean, std = mean[...,None,None], std[...,None,None] 
    imgt = torch.clamp((imgt*std+mean)*255., min=0).numpy().astype(np.uint8)
    return np.transpose(imgt, (1,2,0))
    
    
def show_ex(ds, rows, cols=2, shuffle=True):
    """Display first few examples from dataset"""
    nshow = rows*cols
    fig, axs = plt.subplots(rows, cols, figsize=(16,8))
    for _ in range(nshow):
        i = _ if not shuffle else np.random.randint(len(ds))
        img = np.concatenate((deprocess(ds[i][0][0], 'uav'), deprocess(ds[i][0][1], 'sat')), axis=1)
        axs.flatten()[_].set_title(ds[i][1]);axs.flatten()[_].axis('off')
        axs.flatten()[_].imshow(img)

