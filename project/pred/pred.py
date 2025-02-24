import os
from pathlib import Path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

import torch.backends.cudnn as cudnn
import warnings
from project.utilis.dataset import make_dataloader
from project.utilis.make_class import get_classification_preds
from project.utilis.make_seg import get_seg_preds

warnings.filterwarnings('ignore')
cudnn.benchmark = True
def predict(name=None,image_data=None,mode=None,path=Path(r"project\api\attachment\input"),model_class=None,model_seg=None):
    test_data_folder = path
    testset = make_dataloader(test_data_folder,image_data,mode,name,batch_size=1)
    probability_label,df_classification,time_class = get_classification_preds(model_class, testset)
    df_segmentation,time_seg =get_seg_preds(model_seg,testset,df_classification)
    return probability_label,df_segmentation,time_class+time_seg