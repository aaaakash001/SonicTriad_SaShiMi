import torch
from torch.utils.data import DataLoader
from torchaudio.functional import mu_law_encoding, mu_law_decoding
from scipy.io import wavfile
import os
import streamlit as st
from sashimi import SaShiMi, Embedding, generate_audio_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DURATION = 8

# Load the model
def load_model(model, name, _epoch=None):
    global epoch, device, model_name, model_folder
    model_name = name
    model_folder = "./models/" + name

    epoch_file = os.path.join(model_folder, "train_loss.txt")
    if not os.path.exists(epoch_file):
        print("Model not trained yet!")
        return None

    with open(epoch_file, 'r') as f:
        train = [float(i.strip()) for i in f.readlines()]

    with open(os.path.join(model_folder, "validation_loss.txt"), 'r') as f:
        valid = [float(i.strip()) for i in f.readlines()]

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


# Function to generate audio of specified duration
def generate_audio(duration):
    model.eval()
    with torch.no_grad():
        gen = generate_audio_sample(model, duration * 16000, batch_size=1)
        gen = mu_law_decoding(gen, 256).cpu()

    audio_output = []
    for batch in gen:
        audio_output.append(batch.numpy())

    return audio_output


# Gradio interface
def main():

    st.title("Audio Generation Demo")
    st.write("Enter a duration in seconds to generate an audio sample.")

    duration = st.number_input("Duration (seconds)", value=8)

    if st.button("Generate Audio"):
        audio_output = generate_audio(duration)
        audio_output = np.array(audio_output).flatten()
        st.audio(audio_output, format="audio/wav", rate=16000)


if __name__ == "__main__":
    main()
