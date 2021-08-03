# System libs
import csv

import numpy as np
import scipy.io
import torch
import torchvision.transforms
from PIL import Image

# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule

MODE = "ade20k"
# input_image="input.jpg"
input_image = "ADE_val_00001519.jpg"
output_name = "segmentation.png"
shape = (512, 768)

colors = scipy.io.loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

# Network Builders
net_encoder = ModelBuilder.build_encoder(
    arch='resnet50dilated',
    fc_dim=2048,
    weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
net_decoder = ModelBuilder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
segmentation_module.cuda()

# Load and normalize one image as a singleton tensor batch
pil_to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
        std=[0.229, 0.224, 0.225])  # across a large photo dataset.
])

from omegaconf import OmegaConf

config_path = "./logs/2020-11-06T23-28-03_ade20k_segmentation_16161024/configs/2020-11-20T21-45-44-project.yaml"
config = OmegaConf.load(config_path)
import yaml

print(yaml.dump(OmegaConf.to_container(config)))

from taming.models.cond_transformer import Net2NetTransformer

model = Net2NetTransformer(**config.model.params)

ckpt_path = "./logs/2020-11-06T23-28-03_ade20k_segmentation_16161024/checkpoints/last.ckpt"
sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=False)
model.cuda().eval()
torch.set_grad_enabled(False)


def run_transforming():
    pass


def segmentate_picture(input_img):
    pil_image = Image.open(input_img).convert('RGB')
    img_data = pil_to_tensor(pil_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]

    torch.cuda.empty_cache()

    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)

    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()

    pred = pred.copy()
    pred = list(map(lambda x: x + 1, pred.flatten()))
    pred = np.reshape(pred, shape)
    result = Image.fromarray(np.uint8(pred), 'L')
    return np.asarray(result)


segmentation = segmentate_picture(input_image)

segmentation = np.eye(151)[segmentation]
segmentation = torch.tensor(segmentation.transpose(2, 0, 1)[None]).to(dtype=torch.float32, device=model.device)

c_code, c_indices = model.encode_to_c(segmentation)
print("c_code", c_code.shape, c_code.dtype)
print("c_indices", c_indices.shape, c_indices.dtype)
qwe = c_code.shape[2] * c_code.shape[3]
assert qwe == c_indices.shape[0]
segmentation_rec = model.cond_stage_model.decode(c_code)


def save_result_image(s):
    s = s.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
    s = ((s + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    s = Image.fromarray(s)
    s.save("result.jpg")


codebook_size = config.model.params.first_stage_config.params.embed_dim
z_indices_shape = c_indices.shape
z_code_shape = c_code.shape
z_indices = torch.randint(codebook_size, z_indices_shape, device=model.device)
x_sample = model.decode_to_img(z_indices, z_code_shape)


import time

idx = z_indices
idx = idx.reshape(z_code_shape[0], z_code_shape[2], z_code_shape[3])

cidx = c_indices
cidx = cidx.reshape(c_code.shape[0], c_code.shape[2], c_code.shape[3])

temperature = 1.0
top_k = 100
update_every = 50

win_size = 16
win_size_divved = win_size // 2

start_t = time.time()
for i in range(0, z_code_shape[2] - 0):
    if i <= win_size_divved:
        local_i = i
    elif z_code_shape[2] - i < win_size_divved:
        local_i = win_size - (z_code_shape[2] - i)
    else:
        local_i = win_size_divved
    for j in range(0, z_code_shape[3] - 0):
        if j <= win_size_divved:
            local_j = j
        elif z_code_shape[3] - j < win_size_divved:
            local_j = win_size - (z_code_shape[3] - j)
        else:
            local_j = win_size_divved

        i_start = i - local_i
        i_end = i_start + win_size
        j_start = j - local_j
        j_end = j_start + win_size

        patch = idx[:, i_start:i_end, j_start:j_end]
        patch = patch.reshape(patch.shape[0], -1)
        cpatch = cidx[:, i_start:i_end, j_start:j_end]
        cpatch = cpatch.reshape(cpatch.shape[0], -1)
        patch = torch.cat((cpatch, patch), dim=1)
        logits, _ = model.transformer(patch[:, :-1])
        logits = logits[:, -win_size * win_size:, :]
        logits = logits.reshape(z_code_shape[0], win_size, win_size, -1)
        logits = logits[:, local_i, local_j, :]

        logits = logits / temperature

        if top_k is not None:
            logits = model.top_k_logits(logits, top_k)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx[:, i, j] = torch.multinomial(probs, num_samples=1)

        step = i * z_code_shape[3] + j
        if step % update_every == 0 or step == z_code_shape[2] * z_code_shape[3] - 1:
            x_sample = model.decode_to_img(idx, z_code_shape)
            print(f"Time: {time.time() - start_t} seconds")
            print(f"Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
            save_result_image(x_sample)
