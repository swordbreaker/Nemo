import nemo.collections.tts.models as models
import numpy as np
import time


# Load mel spectrogram generator
model = models.VitsModel
vits = model.load_from_checkpoint(checkpoint_path="exp/VITS/2023-02-17_11-58-59/checkpoints/VITS--loss_gen_all=44.5953-epoch=1436-last.ckpt")
# Generate audio
import soundfile as sf

def syn(text, path, speaker):
    start = time.time()
    tokens = vits.parse(text=text)
    wave = vits(tokens=tokens, speakers=speaker)[0].squeeze(1)
    wav = np.ravel(wave.to('cpu').detach().numpy())

    dur =  (time.time() - start)
    rtf = (time.time() - start) / (len(wav) / 22050)
    print(f"RTF = {rtf:5f}")
    print(f"Dur = {dur:5f}")
    # Save the audio to disk in a file called speech.wav
    sf.write(path, wav, 22050)

syn("Was sölli bloss mache?", "wavs/ag-103784.wav", 0)
syn("Das esch aspruchsvoll au för de Hörer.", "wavs/ag-101472.wav", 0)

syn("Was söu i bloss machä?", "wavs/be-103784.wav", 1)
syn("Das isch aspruchsvou o fürä Hörer.", "wavs/be-101472.wav", 1)

syn("Was söll ich bloss due?", "wavs/bs-103784.wav", 2)
syn("Das isch aspruchsvoll au für de Hörer.", "wavs/bs-101472.wav", 2)

syn("Was söll i numa macha?", "wavs/gr-103784.wav", 3)
syn("Das isch aspruchsvoll au für da Hörer.", "wavs/gr-101472.wav", 3)

syn("Was sell ech bloss mache?", "wavs/lu-103784.wav", 4)
syn("Das isch aspruchsvoll au für de Hörer", "wavs/lu-101472.wav", 4)

syn("Was sell ech bloss mache?", "wavs/sg-103784.wav", 5)
syn("Das isch aspruchsvoll au für de Hörer", "wavs/sg-101472.wav", 5)

syn("Was sell ech bloss mache?", "wavs/vs-103784.wav", 5)
syn("Das isch aspruchsvoll au für de Hörer", "wavs/vs-101472.wav", 5)

syn("Was sell ech bloss mache?", "wavs/zh-103784.wav", 6)
syn("Das isch aspruchsvoll au für de Hörer", "wavs/zh-101472.wav", 6)