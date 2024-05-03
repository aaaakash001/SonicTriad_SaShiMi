import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import copy
from torchaudio.functional import mu_law_encoding, mu_law_decoding
from scipy.io import wavfile
import os
import sys



# Custom modules
from sashimi import *
from wav_dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
device

DURATION = int(sys.argv[1]  if sys.argv[1] is not None else 8)

def load_model(model, name, _epoch=None):
    global epoch, device, model_name, model_folder
    model_name = name
    model_folder = "./models/" + name

    with open(os.path.join(model_folder, "train_loss.txt"), 'r') as f:
        train = [float(i.strip()) for i in f.readlines()]

    with open(os.path.join(model_folder, "validation_loss.txt"), 'r') as f:
        valid = [float(i.strip()) for i in f.readlines()]

    plt.title(label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train)
    plt.plot(valid)

    epoch = len(train) if _epoch is None else _epoch
    checkpoint_filename = os.path.join(model_folder, f"epoch{epoch:04d}.pt")
    print("Loading model:", checkpoint_filename)
    checkpoint = torch.load(checkpoint_filename, map_location=device)

    return model.load_state_dict(checkpoint["model_state"])


model = SaShiMi(
    input_dim=1,
    hidden_dim=64,
    output_dim=256,
    state_dim=64,
    sequence_length=16000*8,
    block_count=8,
    encoder=Embedding(256, 64),
).to(device)

load_model(model, "ym-8l")


def sample(DURATION):

    seed = 42
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        gen = generate_audio_sample(model, DURATION*16000, batch_size=1)
        gen = mu_law_decoding(gen, 256).cpu()

    dirname = "outputs/" + model_name
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for batch in gen:
        wavfile.write(f"{dirname}/generated_sample.wav", 16000, batch.numpy())


    return 

sample(DURATION=DURATION)