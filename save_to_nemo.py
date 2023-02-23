import nemo.collections.tts.models as models
import numpy as np

# Load mel spectrogram generator
model = models.VitsModel
vits = model.load_from_checkpoint(checkpoint_path="exp/VITS/2023-02-17_11-58-59/checkpoints/VITS--loss_gen_all=44.5953-epoch=1436-last.ckpt")

vits.save_to("vits.nemo")