import torch
from torchvision.io import read_image, ImageReadMode
import os
from torchvision import transforms

import segmented_vit_profile

state_dict_filename = "vit_state_dict.pt"
if not os.path.isfile(state_dict_filename):
    segmented_vit_profile.save_base_state_dict(state_dict_filename)
base_state_dict = torch.load(state_dict_filename)

preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

for path in os.listdir("imagenette_samples"):
    if "jpeg" not in path.lower():
        continue

    # run the sample images through the appropriate transform function
    x = read_image(f"imagenette_samples/{path}", ImageReadMode.RGB)
    preprocessed = preprocess(x)

    name = path.replace(".JPEG", "")
    os.makedirs("vit_samples/conv_proj", exist_ok=True)
    preprocessed = preprocessed.unsqueeze(0)
    torch.save(preprocessed, f"vit_samples/conv_proj/{name}.pt")


layer_order = [
    "conv_proj",
    "process_input_operations",
    "add_pos_embedding",
    "encoder_layer_0",
    "encoder_layer_1",
    "encoder_layer_2",
    "encoder_layer_3",
    "encoder_layer_4",
    "encoder_layer_5",
    "encoder_layer_6",
    "encoder_layer_7",
    "encoder_layer_8",
    "encoder_layer_9",
    "encoder_layer_10",
    "encoder_layer_11",
    "head",
]
for i in range(len(layer_order) - 1):
    name = layer_order[i]
    get_layer = getattr(segmented_vit_profile, f"get_{name}")
    layer = get_layer(base_state_dict)

    base_path = f"vit_samples/{name}"
    print(name)
    for t, tensor_path in enumerate(os.listdir(base_path)):
        x = torch.load(f"{base_path}/{tensor_path}")

        y = layer(x)

        if name == "encoder_layer_11":
            y = y[:, 0]

        os.makedirs(f"vit_samples/{layer_order[i+1]}", exist_ok=True)

        torch.save(y, f"vit_samples/{layer_order[i+1]}/{tensor_path}")
