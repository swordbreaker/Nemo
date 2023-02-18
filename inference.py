import nemo.collections.tts.models as models
import numpy as np

# Load mel spectrogram generator
model = models.VitsModel
vits = model.load_from_checkpoint(checkpoint_path="exp/VITS/2023-02-17_11-58-59/checkpoints/VITS--loss_gen_all=25.9032-epoch=80.ckpt")
# Generate audio
import soundfile as sf
tokens = vits.parse(text="Da isch e tescht.")
wave = vits(tokens=tokens, speakers=0)[0].squeeze(1)
# wave = vits.convert_text_to_waveform(tokens=tokens[0], speakers=0)
# Save the audio to disk in a file called speech.wav
sf.write("./speech.wav", np.ravel(wave.to('cpu').detach().numpy()), 22050)