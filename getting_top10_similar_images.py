import os
import sys
import re
import gc
import platform
import random
import matplotlib.pyplot as plt #графики
import plotly.express as px #графики
import seaborn as sns #графики


import numpy as np
import pandas as pd #пандас для загрузки датасета
from tqdm import tqdm # красивое отображение циклов

import torch #пайторч
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import torchvision.transforms as transforms 


from datasets import Dataset # датасет для хаггингфейс
import pandas as pd
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel, BeitForImageClassification # нужно, чтобы качать модели с hf

from sklearn.metrics import roc_auc_score, f1_score # рассчитывать метрики
from sklearn.model_selection import StratifiedKFold, train_test_split
import evaluate  # рассчитывать метрики

import timm # библиотека с нужными моделями
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import glob
import cv2
from PIL import Image

import albumentations as A # аугментации
from albumentations.pytorch import ToTensorV2
from tqdm.contrib import tzip
import warnings
warnings.simplefilter('ignore')
from transformers import XCLIPProcessor, XCLIPModel # трансформеры
from transformers import TrainingArguments, Trainer # обучение трансформеров
from sklearn.model_selection import KFold # кросс валидация
# from lion_pytorch import Lion # оптимайзер

from mpl_toolkits.axes_grid1 import ImageGrid # КРАСИВОЕ отображение фоток
from glob import glob
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm



model = timm.create_model('swinv2_base_window16_256', pretrained=True, num_classes=1)
config = resolve_data_config({}, model=model)
processorr = create_transform(**config)
model.head.fc = model.head.flatten
preprocessor = AutoImageProcessor.from_pretrained("swinv2",  padding=False)
modell = AutoModelForImageClassification.from_pretrained("swinv2", num_labels=15, ignore_mismatched_sizes=True)
modell.to('cuda')



training_args = TrainingArguments(
    output_dir="models/",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=1e-4,
    per_device_train_batch_size=200,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=128,
    num_train_epochs=3,
    logging_steps=10,
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
    optim='adamw_torch',
    save_total_limit=1,
    disable_tqdm=True
    
)
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()
cls = modell
trainer = Trainer(
    data_collator=data_collator,
    model=cls,
    args=training_args,
    tokenizer=preprocessor,

)


def auto_transforms(examples):
    size = 224,224
    images = [Image.open(path).convert("RGB") for path in examples["img"]]
    inputs = preprocessor(images=images, return_tensors="pt")
    inputs['labels'] = torch.tensor(examples['cls'])
    return inputs

def get_cls(path):
    name = path
    ds = Dataset.from_pandas(pd.DataFrame(data={'img':[name],'cls':[0]}))
    ds = ds.with_transform(auto_transforms)
    new_predictions=trainer.predict(ds)
    probabilities = nn.functional.softmax(torch.tensor(new_predictions.predictions[0]), dim=-1)
    cls_name = ['Археология', 'Оружие', 'Прочие', 'Нумизматика', 'Фото, негативы',
       'Редкие книги', 'Документы', 'Печатная продукция', 'ДПИ',
       'Скульптура', 'Графика', 'Техника', 'Живопись',
       'Естественнонауч.коллекция', 'Минералогия']

    return cls_name[probabilities.argmax()], max(probabilities)


def cos(nums1, nums2):
    return np.dot(nums1, nums2) / (np.linalg.norm(nums1) * np.linalg.norm(nums2))

with open('emb.pkl', 'rb') as f:
    dd = pickle.load(f)


def get_10_top(path):
    img1 = Image.open(path).convert('RGB')
    img1 = processorr(img1).unsqueeze(0).to('cuda')
    classs, uv = get_cls(path)
    
    img1 = model(img1.to('cuda'))[0].cpu().detach().numpy()
    top = []

    if uv > 0.84:
        for i in dd[classs]:
            top.append([dot(i[2], img1)/(norm(i[2])*norm(img1)), i[0], i[1]])
    else:
        for i in dd:
            for j in dd[i]:
                top.append([dot(j[2], img1)/(norm(j[2])*norm(img1)), j[0], j[1]])
    top = sorted(top)
    return top[-10:]