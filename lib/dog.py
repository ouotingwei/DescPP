import numpy as np
import pycolmap

class Dog:
    def __init__(self, nfeatures=-1, patch_size=32, mr_size=12):
        self.nfeatures = nfeatures
        self.patch_size = patch_size
        self.mr_size = mr_size
        
        use_gpu = pycolmap.has_cuda
        options = {
            'first_octave': 0,
            'peak_threshold': 0.01,
            'normalization': pycolmap.Normalization.L2
        }

        self.sift = pycolmap.Sift(
            options=pycolmap.SiftExtractionOptions(options),
            device=getattr(pycolmap.Device, 'cuda' if use_gpu else 'cpu'))
    
    def detectAndCompute(self, img):
        keypoints, descriptors = self.sift.extract(img)

        if self.nfeatures != -1 and keypoints.shape[0] > self.nfeatures:
            keypoints = keypoints[:self.nfeatures]
            descriptors = descriptors[:self.nfeatures]

        scores = None

        return keypoints, scores, descriptors