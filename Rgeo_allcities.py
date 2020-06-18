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
# retrieve the labels file
with h5py.File('data_labels.h5', 'r') as f:
    print (list(f.keys()))
    match_array = np.asarray(list(f['match_array_40']))
def get_match_array(city):
    global match_array
    start_i = {'atlanta'  :0,
            'austin'      :2078,
            'boston'      :3075,
            'champaign'   :4272,
            'chicago'     :11466,
            'miami'       :14581,
            'sanfrancisco':16167,
            'springfield' :17059,
            'stlouis'     :23599}
    end_i   = {'atlanta'  :2078,
            'austin'      :3075,
            'boston'      :4272,
            'champaign'   :11466,
            'chicago'     :14581,
            'miami'       :16167,
            'sanfrancisco':17059,
            'springfield' :23599,
            'stlouis'     :30512}
    return match_array[start_i[city]:end_i[city], start_i[city]:end_i[city]]
    
################################################################################################################
# generate matching and non-matching pairs
def get_pairs(city):
    match_array = get_match_array(city)
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
# class to get paths of uav and satellite images with labels
class AerialCity():
    def __init__(self, city):
        self.city     = city
        self.sat_img_dir = 'train/' + self.city + '/sat-small'
        self.uav_img_dir = 'train/' + self.city + '/uav-small'
        
    def get_aerial_images(self):
        '''List of all the images, sorted by `natural_keys`'''
        fuav = get_image_files(self.uav_img_dir)
        fsat = get_image_files(self.sat_img_dir)
        fuav = create_sorted_list(fuav)
        fsat = create_sorted_list(fsat)
        return fuav, fsat

    def get_info(self):
        matching_pairs, non_match_pairs = get_pairs(self.city) #todo: get pairs should work for all cities
        fuav, fsat = self.get_aerial_images()
        uav_pathsdf = [fuav[idx] for idx in matching_pairs[:,0]] + [fuav[idx] for idx in non_match_pairs[:,0]]
        sat_pathsdf = [fsat[idx] for idx in matching_pairs[:,1]] + [fsat[idx] for idx in non_match_pairs[:,1]]
        n_matches = len(matching_pairs)
        assert n_matches == len(non_match_pairs)
        labeldf = [1] * n_matches + [0] * n_matches
        is_validdf = ([False] *(4*n_matches//5) + [True] *(n_matches - 4*n_matches//5))*2
        return (uav_pathsdf,
                sat_pathsdf,
                labeldf,
                is_validdf)
        

################################################################################################################
# class to get a dataframe with paths of uav and satellite images with labels
class AerialCities():
    def __init__(self, cities):
        self.cities   = cities
        self.uav_pathsdf = []
        self.sat_pathsdf = []
        self.labeldf     = []
        self.is_validdf  = []
    
    def create_df(self):
        for i in range(len(self.cities)):
            city = AerialCity(self.cities[i])
            l1, l2, l3, l4 = city.get_info()
            self.uav_pathsdf.extend(l1)
            self.sat_pathsdf.extend(l2)
            self.labeldf    .extend(l3)
            self.is_validdf .extend(l4)
        dicdf = {'uav'    : self.uav_pathsdf,
                'sat'     : self.sat_pathsdf,
                'label'   : self.labeldf,
                'is_valid': self.is_validdf}
        df = pd.DataFrame(dicdf)
        return df   

################################################################################################################
################################################################################################################
# custom itemlist
mean, std = torch.tensor(imagenet_stats)
# for 3 channels
mean = mean[...,None,None]
std = std  [...,None,None]
# The primary difference from the tutorial is with how normalization is being done here
class ImageTuple(ItemBase):
    def __init__(self, img1, img2):
        self.img1,self.img2 = img1,img2
        self.obj = (img1,img2)
        self.data = [(img1.data-mean)/std, (img2.data-mean)/std]
    def apply_tfms(self, tfms, **kwargs):
        self.img1 = self.img1.apply_tfms(tfms[0], **kwargs)
        self.img2 = self.img2.apply_tfms(tfms[1], **kwargs)
        self.data = [(self.img1.data-mean)/std, (self.img2.data-mean)/std]
        return self    
    def to_one(self): return Image(mean+torch.cat(self.data,2)*std)
    def __repr__(self): return f'{self.__class__.__name__} {self.img1.shape, self.img2.shape}'

class ImageTupleList(ImageList):
    def __init__(self, items, itemsB=None, **kwargs):
        super().__init__(items, **kwargs)
        self.itemsB = itemsB
        self.copy_new.append('itemsB')
    def get(self, i):
        img1 = super().get(i)
        fn = self.itemsB[i]
        return ImageTuple(img1, open_image(fn))
    
    @classmethod
    def from_dfs(cls, df:DataFrame, path='.', cols=0, colsB=1, **kwargs):
        "Create an `ItemList` in `path` from the inputs in the `cols` of `df`."
        t_itemsB = ImageList.from_df(df[df['is_valid']==False], path, colsB).items
        t_res = super().from_df(df[df['is_valid']==False],path,cols, itemsB=t_itemsB, **kwargs)
        v_itemsB = ImageList.from_df(df[df['is_valid']==True],  path, colsB).items
        v_res = super().from_df(df[df['is_valid']==True],path,cols, itemsB=v_itemsB, **kwargs)
        t_res.path = v_res.path = path
        return ItemLists(t_res.path, t_res, v_res)
       
    def reconstruct(self, t:Tensor): 
        return ImageTuple(Image(t[0]*std+mean),Image(t[1]*std+mean))
    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(12,6), **kwargs):
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, y=ys[i], **kwargs)
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, figsize:Tuple[int,int]=(12,6), **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
        `kwargs` are passed to the show method."""
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
        for i,(ax,x,y,z) in enumerate(zip(axs.flatten(),xs,ys,zs)):
            x.to_one().show(ax=ax, title = str(y)+', '+str(z), **kwargs)

################################################################################################################
# random 90 degree rotations transform
def _rot90_affine(k:partial(uniform_int, 0, 3)):
# adapted from kechan's implementation, see fastai/fastai issue #1653
    "Randomly rotate `x` image based on `k` as in np.rot90"
    #print("k={}".format(k))
    if k%2 == 0:
        x = -1. if k&2 else 1.
        y = -1. if k&2 else 1.
        
        return [[x, 0, 0.],
                [0, y, 0],
                [0, 0, 1.]]
    else:
        x = 1. if k&2 else -1.
        y = -1. if k&2 else 1.
        
        return [[0, x, 0.],
                [y, 0, 0],
                [0, 0, 1.]]

rot90_affine = RandTransform(tfm=TfmAffine(_rot90_affine),
                            kwargs={}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True)

################################################################################################################
# contrastive loss with margin = 100
# contrastive loss with margin = 100
def loss_contrastive(dist, label, margin=100, reduction='mean'):
    loss = (label) * torch.pow(dist, 2) + (1-label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    if reduction == 'mean' : return loss.mean()
    elif reduction == 'sum': return loss.sum()
    else                   : return loss
class ContrastiveLoss(nn.Module):    
    def __init__(self, margin=100, reduction='mean', **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    def forward(self, dist, label, **kwargs):
        return loss_contrastive(dist, label, self.margin, self.reduction)
    
# cross entropy loss for this problem
def loss_ce(out, targ):
    return CrossEntropyFlat()(out, targ.long())
################################################################################################################
# accuracy metric according to the set margin
def accuracy_contrastive(input:Tensor, targs:Tensor)->Rank0Tensor:
    "Computes accuracy with `targs` when `input` is bs"
    input = input<50
    return (input==targs).float().mean()

################################################################################################################
# make a histogram of the distances from results of inference
def histo(preds, dset):
    matdist, nmatdist = [], []
    for i in range(len(preds[0])):
        if preds[1][i] == 0: nmatdist.append(preds[0][i].numpy())
        if preds[1][i] == 1: matdist. append(preds[0][i].numpy())

    mathist  = plt.hist( matdist, range(int(np.ceil(max( matdist)))), label = 'matching pairs', alpha = 0.7)
    nmathist = plt.hist(nmatdist, range(int(np.ceil(max(nmatdist)))), label = 'non-matching pairs', alpha = 0.7)
    plt.legend(loc='best')
    plt.title(dset + ' data')
    plt.xlabel('Distance between image pairs')
    plt.show()            