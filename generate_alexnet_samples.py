import torch
from torchvision.io import read_image, ImageReadMode
import os
from torchvision import transforms

import segmented_alexnet_profile

state_dict_filename = "alexnet_state_dict.pt"
if not os.path.isfile(state_dict_filename):
    segmented_alexnet_profile.save_base_state_dict(state_dict_filename)

base_state_dict = torch.load(state_dict_filename)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for path in os.listdir("imagenette_samples"):
    if "jpeg" not in path.lower():
        continue

    # run the sample images through the appropriate transform function
    x = read_image(f"imagenette_samples/{path}", ImageReadMode.RGB)
    preprocessed = preprocess(x)

    name = path.replace(".JPEG", "")
    os.makedirs("alexnet_samples/conv1", exist_ok=True)

    torch.save(preprocessed, f"alexnet_samples/conv1/{name}.pt")


layer_order = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "fc3"]
for i in range(len(layer_order) - 1):
    name = layer_order[i]
    get_layer = getattr(segmented_alexnet_profile, f"get_{name}")
    layer = get_layer(base_state_dict)

    base_path = f"alexnet_samples/{name}"
    for tensor_path in os.listdir(base_path):
        x = torch.load(f"{base_path}/{tensor_path}")
        y = layer(x)
        if "conv5" in name:
            y = torch.flatten(y, start_dim=0, end_dim=-1)

        os.makedirs(f"alexnet_samples/{layer_order[i+1]}", exist_ok=True)

        torch.save(y, f"alexnet_samples/{layer_order[i+1]}/{tensor_path}")
