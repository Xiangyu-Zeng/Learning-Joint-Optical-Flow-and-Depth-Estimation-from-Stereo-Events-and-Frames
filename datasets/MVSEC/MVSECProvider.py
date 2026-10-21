import os
import torch.utils.data

from .MVSECSequence_flow import MVSECSequence_flow
from .MVSECSequence_disparity import MVSECSequence_disparity
from .MVSECSequence_flow_with_image import MVSECSequence_flow_with_image

class MVSECProvider:
    def __init__(self, cfgs):
        self.class_dict = {'MVSECSequence_flow':MVSECSequence_flow,
                           'MVSECSequence_disparity':MVSECSequence_disparity,
                           'MVSECSequence_flow_with_image':MVSECSequence_flow_with_image}


        self.SequenceClass = self.class_dict[cfgs['SequenceClass']]


        self.MVSECSequences = []
        self.SequenceNames = cfgs['SequenceNames']
        for name in self.SequenceNames:
            self.MVSECSequences.append(self.SequenceClass(cfgs, name))

    def get_datasets(self, F_list=False):
        if F_list:
            return self.MVSECSequences, self.SequenceNames
        else:
            return torch.utils.data.ConcatDataset(self.MVSECSequences)
