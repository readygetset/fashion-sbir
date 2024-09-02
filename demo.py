import sys
import torch
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
import gradio as gr

sys.path.append('../code/')
from clip.model import CLIP
from clip.clip import _transform, tokenize

# Load Model
gpus = [0, 1, 2, 3, 4]
torch.cuda.set_device(gpus[0])

model_config_file = "../code/training/model_configs/ViT-B-16.json"
model_file = "../ckpt/taskformer_model.pth"

with open(model_config_file, 'r') as f:
    model_info = json.load(f)
        
model = CLIP(**model_info)

loc = f"cuda:{gpus[0]}"
checkpoint = torch.load(model_file, map_location=loc)

if next(iter(checkpoint.items()))[0].startswith('module'):
    checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
    
model.load_state_dict(checkpoint, strict=False)
model = torch.nn.DataParallel(model, device_ids=gpus)
model = model.cuda().eval()

# Data Preprocessing
def preprocess_images(image_list, transform):
    dataset = SimpleImageFolder(image_list, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader

class SimpleImageFolder(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as img:
            if self.transform is not None:
                img = self.transform(img)
            return img, image_path

    def __len__(self):
        return len(self.image_paths)

def get_image_paths(*dirs):
    image_list = []
    for path in dirs:
        for item in Path(path).glob('*'):
            if '.ipynb' not in str(item):
                image_list.append(str(item))
    return image_list

preprocess_val = _transform(model.module.visual.input_resolution, is_train=False)
image_list = get_image_paths('../../dataset/karol-skorulski/karol-skorulski_image', 
                             '../../dataset/ghoumrassi/ghoumrassi_image')
dataloader = preprocess_images(image_list, preprocess_val)

def extract_features(model, dataloader):
    all_image_features, all_image_paths = [], []
    with torch.no_grad():
        for images, image_paths in dataloader:
            images = images.cuda(non_blocking=True)
            image_features = model.module.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features.cpu().numpy())
            all_image_paths.extend(image_paths)
    return np.vstack(all_image_features), all_image_paths

all_image_features, all_image_paths = extract_features(model, dataloader)

def get_feature(model, sketch, text, transformer):
    img_tensor = transformer(sketch).unsqueeze(0).cuda()
    txt_tensor = tokenize([str(text)]).cuda()
    with torch.no_grad():
        sketch_feature = model.module.encode_sketch(img_tensor)
        text_feature = model.module.encode_text(txt_tensor)
        sketch_feature /= sketch_feature.norm(dim=-1, keepdim=True)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
    return model.module.feature_fuse(sketch_feature, text_feature)

nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine').fit(all_image_features)

def mark_boundary(img, color=(0, 255, 0)):
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 5, img.width - 5, img.height - 5], outline=color, width=10)
    return img

def get_concat_hn(ims):
    sum_w = len(ims) * 256
    dst = Image.new('RGB', (sum_w, 256))
    for i, im in enumerate(ims):
        dst.paste(im.resize((256, 256)), (i * 256, 0))
    return dst

def get_image_list(query_feat, nbrs, all_image_paths):
    distances, indices = nbrs.kneighbors(query_feat.cpu().numpy())
    im_list = []
    for ind in indices[0]:
        file_loc = all_image_paths[ind]
        with Image.open(file_loc) as img:
            img = img.convert("RGB")
            im_list.append(img)
    return im_list

# Gradio 인터페이스 함수
def generate_results(sketch, caption):
    query_feat = get_feature(model, sketch, caption, preprocess_val)
    im_list = [sketch] + get_image_list(query_feat, nbrs, all_image_paths)
    result_image = get_concat_hn(im_list)
    return result_image

# Gradio Interface Setup
sketch_input = gr.Sketchpad(label="Draw your sketch here")
caption_input = gr.Textbox(lines=2, placeholder="Enter caption here...", label="Caption")

gr.Interface(fn=generate_results, 
             inputs=[sketch_input, caption_input], 
              outputs=gr.Image(type="pil"),
             title="Sketch & Caption-based Image Retrieval",
             description="Draw a sketch and provide a caption to retrieve similar images from the dataset."
            ).launch(share=True)