from absl import flags, app
from detect_train import Custom_Model

import torch
import torch.nn as nn
import numpy as np

import os

FLAGS = flags.FLAGS

flags.DEFINE_string("files", default="/home/m_bobrin/goznak/Task2Noise_Speech/Goznak_ML_Tasks/train1/train/train/clean/29", help="Path to directory with mel spectrogram files.")
flags.DEFINE_string("saved_model", default="/home/m_bobrin/goznak/Task2Noise_Speech/notebooks/custom_model.pth", help="Path to pretrained model.")

class Inference:
    def __init__(self, model: nn.Module, path_to_model: str, data: str):
        
        self.model.load_state_dict(torch.load(path_to_model)).cuda()
        self.model.eval()
        
        self.data = data
        
    def inference(self):
        files = os.listdir(self.data)
        
        mel_specs = [torch.from_numpy(self._process(np.load(files[i]).T)[None, None, :, :]).cuda() for i in files]
        #processed_sample = torch.Tensor(processed_sample).unsqueeze(0)
        
        for mel in mel_specs:
            
        
    @staticmethod
    def _process(sample: np.ndarray) -> np.ndarray:
        '''
        Method for truncating/shifting sample
        '''
        
        if sample.shape[1] < 700:
            processed_sample = np.pad(sample, ((0, 0), (0, 700 - sample.shape[1])))
        else:
            processed_sample = sample[:, :700]
        return processed_sample
        
def main(_):
    
    custom_model = Custom_Model()
    inf = Inference(FLAGS.saved_model, FLAGS.files)






if __name__ == "__main__":
    app.run(main)