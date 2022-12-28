from absl import app, flags

from dataset import ClassificationSound

from tqdm.auto import tqdm
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from initialize_weights import xavier_init, kaiming_init

from sklearn.metrics import accuracy_score

# Instead of usual Argparse due to existing problems
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "speech_dir_train",
    default="/home/m_bobrin/goznak/Task2Noise_Speech/Goznak_ML_Tasks/train1/train/train/",
    help="Path to noisy and clean folders to train on.",
)
flags.DEFINE_string("speech_dir_valid", default='/home/m_bobrin/goznak/Task2Noise_Speech/Goznak_ML_Tasks/val/val/val/', help="Path to validation mel specs.")
flags.DEFINE_integer("num_epochs", default=5, help="Number of epochs of training.")
flags.DEFINE_boolean("use_cuda", default=True, help="Whether to use cuda.")


class Custom_Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.custom_model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), bias=True),
            nn.LeakyReLU(),
            
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2), bias=True),
            
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=5, stride=3),
            
            nn.Flatten(),
            nn.Linear(in_features=864, out_features=1, bias=True)
        )
    
    def forward(self, x):
        return self.custom_model(x)

def run_train(custom_model, ds_speech_loader_train, ds_speech_loader_val):
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=3e-4)

    num_epochs = FLAGS.num_epochs
    loss_fn = nn.BCEWithLogitsLoss()

    acc = []
    train_loss = []

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(ds_speech_loader_train)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    
    plotting = 0
    for epoch in range(num_epochs):
        for x, y in tqdm(ds_speech_loader_train, leave=True, position=0):
            
            custom_model.train()
            x = x.cuda()
            y = y.cuda()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                predict = custom_model(x)
                loss = loss_fn(predict.squeeze(), y)
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            #print(f"Loss on current batch: {loss.item()}")
            train_loss.append(loss.item())
            
            acc_train = torch.where(predict > 0.5, 1, 0).squeeze()
            acc_train_metric = accuracy_score(y.detach().cpu().numpy(), acc_train.detach().cpu().numpy().ravel())
            
            print(f'Epoch: {epoch}, Loss on current batch: {loss.item():.2f}, Accuracy Train batch: {acc_train_metric:.2f}', end='\r')
            with torch.no_grad():
                for x_val, y_val in ds_speech_loader_val:
                    with torch.cuda.amp.autocast():
                        x_val = x_val.cuda()
                        y_val = y_val.cuda()
                        val_predict = custom_model(x_val)
                        val_predict = torch.where(val_predict > 0.5, 1, 0).squeeze()
            
            plotting += 1
            if plotting % 30 == 0:
                acc_valid = accuracy_score(y_val.detach().cpu().numpy(), val_predict.detach().cpu().numpy().ravel())
                acc.append(acc_valid)
                print(f"Accuracy Validation: {acc_valid}")
                plt.figure(figsize=(10,5))
                plt.subplot(121)
                plt.semilogy(train_loss)
                plt.title('Train loss')
                plt.grid()
                plt.subplot(122)
                plt.plot(acc)
                plt.title('val Accuracy')
                plt.grid()
                plt.savefig(f"./assets/Step_{plotting}_epoch_{epoch}.jpg")

def main(_):
    """
    Entry point for training classification task
    """

    ds_speech_train = ClassificationSound(FLAGS.speech_dir_train)
    ds_speech_loader_train = DataLoader(ds_speech_train, batch_size=256, shuffle=True)
    
    ds_speech_val = ClassificationSound(FLAGS.speech_dir_valid)
    s_speech_loader_val = DataLoader(ds_speech_val, batch_size=256, shuffle=True)
    
    custom_model = Custom_Model()
    if FLAGS.use_cuda:
        custom_model = custom_model.cuda()
        
    print("Running training")
    run_train(custom_model, ds_speech_loader_train, s_speech_loader_val)
    print("Saving model weights")
    
    torch.save(custom_model.state_dict(), "custom_model.pth")
    
if __name__ == "__main__":
    app.run(main)
