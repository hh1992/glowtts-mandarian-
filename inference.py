import matplotlib.pyplot as plt
import numpy as np
#import pinyin
from scipy.io.wavfile import write
import sys
import soundfile as sf
import librosa
import numpy as np
import os
import glob
import json
from stft import STFT
import torch
from utils import text_to_sequence

import commons
import attentions
import modules
import models
import utils
import config
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
class Denoiser(torch.nn.Module):
    """ Removes model bias from audio produced with waveglow """

    def __init__(self, waveglow, filter_length=1024, n_overlap=4,
                 win_length=1024, mode='zeros'):
        super(Denoiser, self).__init__()
        self.stft = STFT(filter_length=filter_length,
                         hop_length=int(filter_length / n_overlap),
                         win_length=win_length).cuda()
        if mode == 'zeros':
            mel_input = torch.zeros(
                (1, 80, 88),
                dtype=waveglow.upsample.weight.dtype,
                device=waveglow.upsample.weight.device)
        elif mode == 'normal':
            mel_input = torch.randn(
                (1, 80, 88),
                dtype=waveglow.upsample.weight.dtype,
                device=waveglow.upsample.weight.device)
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            bias_audio = waveglow.infer(mel_input, sigma=0.0).float()
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised

# load waveglow
waveglow_path = 'waveglow_256channels_ljs_v2.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
        k.float()
denoiser = Denoiser(waveglow)
model_dir =  "./checkpoints/"
model= models.FlowGenerator(n_vocab= config.n_symbols, hidden_channels=config.hidden_channels,filter_channels=config.filter_channels,filter_channels_dp=config.filter_channels_dp,out_channels=config.n_mel_channels).to(device)
checkpoint = torch.load(os.path.join(config.model_dir, "G_{}.pth".format(110)),map_location='cuda:0')
model.load_state_dict(checkpoint['model'])
model.decoder.store_inverse()
model.eval()

#text = "相对论直接和间接的催生了量子力学的诞生 也为研究微观世界的高速运动确立了全新的数学模型"
#text = pinyin.get(text, format="numerical", delimiter=" ")
text = "zun1 yi4 shi4 de5 tian1 qi4 jiu4 xiang4 gu1 niang2 yi2 yang4"
sequence = np.array(text_to_sequence(text))[None, :]
x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()
with torch.no_grad():
  noise_scale = .0667
  length_scale = 1.0
  (y_gen_tst, *r), attn_gen, *_ = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)
  mel = y_gen_tst.cpu().detach().numpy()
  print(np.max(mel), np.min(mel))
  try:
        audio = waveglow.infer(y_gen_tst.half(), sigma=.666)
  except:
        audio = waveglow.infer(y_gen_tst, sigma=.666)
audio_denoised = denoiser(audio, strength=0.1)
audio_denoised = audio_denoised[0].data.cpu().numpy()  
audio_denoised = audio_denoised.astype(np.float32)

sf.write('output01.wav', audio_denoised, config.sampling_rate, 'PCM_24')
