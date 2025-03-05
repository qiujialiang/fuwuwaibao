from io import BytesIO
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image
import cv2
plt.ioff()


def name_and_mask(df):
    img_names = [str(i).split('_')[0] for i in df.iloc[0:4, 0].values]
    labels = df.iloc[0:4, 1]
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)
    for idx, label in enumerate(labels.values):
        if not pd.isna(label):  # 检查是否为 NaN
            mask_label = np.zeros(1600 * 256, dtype=np.uint8)
            label = label.split(" ")
            # 过滤空字符串
            label = [item for item in label if item.strip()]
            pos = map(int, label[0::2])
            lengths = map(int, label[1::2]) 
            for p, l in zip(pos, lengths):
                mask_label[p-1:p+l-1] = 1
            mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')
    return img_names[0], mask


def eda(df,data=None,path=None):
    """Exploratory data analysis
    Args:
        df: Segmentation DataFrame
        data: Figure binary data
        path: Path of raw figure
    Return:
        binary_data: Binary data of result figure
    """
    df['ImageId'] = df['ImageId_ClassId'].apply(lambda x:x.split('_')[0])
    df['ClassId'] = df['ImageId_ClassId'].apply(lambda x:x.split('_')[1])
    df['hashMask'] = ~df['EncodedPixels'].isnull()
    mask_count_df = df.groupby('ImageId').agg(np.sum).reset_index()
    mask_count_df.sort_values('hashMask',ascending=False,inplace=True)
    palet=[(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]

    name,mask=name_and_mask(df)
    if data is not None:
        stream=io.BytesIO(data)
        img=Image.open(stream)
        img = np.array(img)
    elif path is not None:
        img=cv2.imread(str(path))
        img=np.array(img)
    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, palet[ch], 2)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # out_path=Path(r"project\api\attachment\output").joinpath(name)
    # cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    _, img_encoded = cv2.imencode('.jpg', img)
    binary_data = img_encoded.tobytes()
    
    #path=Path(r"project\api\attachment\output").joinpath(name+'.png')
    return binary_data
