import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import random
from skimage.transform import resize
import bisect

class MRIBase(Dataset):
    def __init__(self,
                txt_file,
                data_root,
                channels,
                k,
                stage,
                size=256,
                is_test=False,
                random_sample=True
                ):
        self.data_root = data_root
        self.channels = channels
        self.is_test = is_test
        self.random_sample = random_sample
        self.k = k
        if txt_file is None:
            if stage in ['stage1', 'stage2', 'VAE']:
                self.image_paths = [f for f in os.listdir(self.data_root) if
                                    (f.endswith('_t2.nii.gz') or f.endswith('-T2.nii.gz') or f.endswith('_t2.nii'))]
            else:
                self.image_paths = [f for f in os.listdir(self.data_root) if ((f.endswith('_t2.nii.gz') or f.endswith('-T2.nii.gz') or f.endswith('_t2.nii')) and (f[-11:-7] in k))]

            print(self.image_paths)
            self.seg_root = self.data_root.replace('t2', 'seg')
            self.seg_paths = []
            if is_test and os.path.exists(self.seg_root):
                self.seg_paths = [s.replace('t2', 'seg') for s in self.image_paths]
        self.size = size
        self._length = len(self.image_paths)
        self.hash_table = []
        if not self.is_test and not self.random_sample:
            self.prepare_train_data()
        
    def prepare_train_data(self):
        self.hash_table.append(-1)
        cnt = 0
        for ind in range(len(self.image_paths)):
            file_path = os.path.join(self.data_root, self.image_paths[ind])
            image = nib.load(file_path).get_fdata()
            image_shape = image.shape
            cnt += image_shape[-1]
            self.hash_table.append(cnt - 1)
        self._length = cnt

    def __len__(self):
        return self._length

    def __getitem__(self, ind):
        if not self.is_test:
            if self.random_sample:
                file_path = os.path.join(self.data_root, self.image_paths[ind])
                image = nib.load(file_path).get_fdata()
                select_ind = random.randint(0, image.shape[-1]-1)
            else:
                ind_hash = bisect.bisect_left(self.hash_table, ind) - 1
                file_path = os.path.join(self.data_root, self.image_paths[ind_hash])
                image = nib.load(file_path).get_fdata()
                select_ind = ind - self.hash_table[ind_hash] - 1
            image_shape = image.shape
            image = image[..., select_ind:select_ind+1]
        else:
            
            file_path = os.path.join(self.data_root, self.image_paths[ind])
            image = nib.load(file_path).get_fdata()
            image_shape = image.shape
            select_ind = -1

        if self.is_test and len(self.seg_paths)>0:
            seg = nib.load(os.path.join(self.seg_root, self.seg_paths[ind])).get_fdata()
        example = dict({})
        image = resize(image, (self.size, self.size, image.shape[-1]))
        example["image"] = self.mri_norm(image)

        if self.is_test and len(self.seg_paths)>0:
            seg = resize(seg, (self.size, self.size, seg.shape[-1]))
            seg = self.mri_norm(seg, devi=False)
            example["seg"] = seg

        example["select_index"] = select_ind
        example["file_path"] = file_path
        return example

    def mri_norm(self, image, devi=True):
        image = np.array(image).astype(np.float32).clip(min=0)
        H, W, C = image.shape
        image_rs = image.reshape(-1, C)
        image_nm = (image_rs - image_rs.min(0)) / (image_rs.max(0) - image_rs.min(0) + 1e-8)
        image_nm = image_nm.reshape(H, W, C)
        if devi:
            image_nm = image_nm * 2 - 1.0
        return image_nm

class MRIIXITrain(MRIBase):
    def __init__(self, data_root,k, stage, **kwargs):
        super().__init__(txt_file=None,
                        data_root=data_root,
                        channels=130,
                        k = k,
                        stage = stage,
                        **kwargs)

class MRIIXIValidation(MRIBase):
    def __init__(self, data_root, k, stage, **kwargs):
        super().__init__(txt_file=None,
                        data_root=data_root,
                        channels=130,
                        k=k,
                        stage=stage,
                        **kwargs)

class MRITest(MRIBase):
    def __init__(self, data_root, k, stage, **kwargs):
        super().__init__(
                        txt_file=None,
                        data_root=data_root,
                        channels=155,
                        k=k,
                        stage=stage,
                        **kwargs)

