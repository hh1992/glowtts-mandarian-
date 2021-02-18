import matplotlib.pyplot as plt
import numpy as np
import pinyin
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
from audio_processing import griffin_lim
import commons
import attentions
import modules
import models
import utils
import config
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

# load waveglow
waveglow_path = 'waveglow_256channels_ljs_v2.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
        k.float()
model_dir =  "./checkpoints/"
model= models.FlowGenerator(n_vocab= config.n_symbols, hidden_channels=config.hidden_channels,filter_channels=config.filter_channels,filter_channels_dp=config.filter_channels_dp,out_channels=config.n_mel_channels).to(device)
checkpoint = torch.load(os.path.join(config.model_dir, "G_{}.pth".format(110)),map_location='cuda:0')
model.load_state_dict(checkpoint['model'])
model.decoder.store_inverse()
model.eval()

text = "相对论直接和间接的催生了量子力学的诞生 也为研究微观世界的高速运动确立了全新的数学模型"
text = pinyin.get(text, format="numerical", delimiter=" ")
text = "zun1 yi4 shi4 de5 tian1 qi4 jiu4 xiang4 gu1 niang2 yi2 yang4"
sequence = np.array(text_to_sequence(text))[None, :]
x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()

stft = commons.TacotronSTFT(config.filter_length, config.hop_length, config.win_length,
    config.n_mel_channels, config.sampling_rate, config.mel_fmin, config.mel_fmax)

with torch.no_grad():
  noise_scale = .0667
  length_scale = 1.0
  (y_gen_tst, *r), attn_gen, *_ = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)
  mel = y_gen_tst.cpu().detach().numpy()
  #mel = mel[0]
  print(np.shape(mel))
  linear = stft._mel_to_linear(mel)
  print('linear', np.shape(linear))
  audio = griffin_lim(linear, stft.stft_fn)
  audio = audio.cpu().detach().numpy()[0]
  print("audio",np.max(audio), np.min(audio))
  audio/= np.max(np.abs(audio))
  #try:
  #      audio = waveglow.infer(y_gen_tst.half(), sigma=.666)
  #except:
  #      audio = waveglow.infer(y_gen_tst, sigma=.666)

sf.write('output01_gf.wav', audio, config.sampling_rate, 'PCM_24')
