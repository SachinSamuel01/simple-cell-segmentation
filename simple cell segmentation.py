import torch
import torch.nn as nn
from torch.optim import optimizer
from torchvision import transforms as tr
from torch.utils.data import DataLoader as DL, Dataset
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


#Create a model class that inherits nn.module


class Model2(nn.Module):
    def __init__(self, in_f=1, out_f=1, dropout_rate=0.7):
        super(Model2, self).__init__()
        
        # Encoder
        self.enc_conv1 = nn.Sequential(nn.Conv2d(in_f, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.enc_conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.enc_conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # Bottleneck
        self.bottleneck_conv = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        # Decoder
        self.dec_conv3 = nn.Sequential(nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.dec_conv2 = nn.Sequential(nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.dec_conv1 = nn.Sequential(nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final Output
        self.final_conv = nn.Conv2d(64, out_f, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        x = self.pool(enc1)
        x = self.dropout(x)

        enc2 = self.enc_conv2(x)
        x = self.pool(enc2)
        x = self.dropout(x)

        enc3 = self.enc_conv3(x)
        x = self.pool(enc3)
        x = self.dropout(x)

        # Bottleneck
        x = self.bottleneck_conv(x)

        # Decoder
        x = self.upsample(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec_conv3(x)

        x = self.upsample(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec_conv2(x)

        x = self.upsample(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec_conv1(x)

        # Final Output
        x = self.final_conv(x)
        return x








# Define a dataset class
class ImgDataset(Dataset):

  
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
        self.mask_paths = [os.path.join(mask_dir, file) for file in os.listdir(mask_dir)]

        if self.transform is None:
          self.transform = tr.Compose([
            #tr.RandomHorizontalFlip(p=0.5),
            tr.Resize((512, 512)),
            #tr.RandomRotation(degrees=(-15, 15)),
            tr.Grayscale(),
            # Convert to tensor
            tr.ToTensor(),
            # Convert to float
            tr.ConvertImageDtype(torch.float),
          ])

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        
        mask = Image.open(self.mask_paths[index])

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def __len__(self):
        return len(self.image_paths)


# defining early stopping

class EarlyStopping:
    def __init__(self, patience=0, delta=0):
        
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        
        
        torch.save(model.state_dict(), "cell_segmentation_unet.pth")
        self.val_loss_min = val_loss



#tr = tr.Compose([tr.Resize((512, 512)),tr.Grayscale(),tr.ToTensor(),tr.ConvertImageDtype(torch.float),])
# defining data augmentation function


learning_rate = 0.001
epochs = 5
batch_size = 1

# Create training and validation datasets
train_path='D:\\projects\\VS\\project\\simple projects\\simple cell segmentation\\simple cell segmentation\\dataset-cell-segmentation\\archive\\BBBC005_v1_images\\BBBC005_v1_images'
mtrain_path='D:\\projects\\VS\\project\\simple projects\\simple cell segmentation\\simple cell segmentation\\dataset-cell-segmentation\\archive\\BBBC005_v1_ground_truth\\BBBC005_v1_ground_truth'
train_data1 = ImgDataset(train_path, mtrain_path)



#val_data = ImgDataset(val_path, mval_path)
indices=list(range(1200))


train_data = torch.utils.data.Subset(train_data1, indices[:1000])

val_data = torch.utils.data.Subset(train_data1, indices[1000:1200])


train_loader = DL(train_data, batch_size=batch_size, shuffle=True)
val_loader = DL(val_data, batch_size=batch_size)

model=Model2()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define loss function
criterion = nn.BCEWithLogitsLoss()

def iou_val_dice_val(pred,target):
   pred=torch.sigmoid(pred)
   pred=(pred>0.5).float()
   pred=pred.view(-1)
   target = target.view(-1)

   
   intersection = (pred * target).sum()
   total = (pred + target).sum()
   union = total - intersection 
   total = pred.sum() + target.sum()
   IoU = (intersection + 1e-6) / (union + 1e-6)
   dice = (2. * intersection + 1e-6) / (total + 1e-6)
    
   return [IoU.item(),dice.item()]


def cal_precision_recall(pred,target):
   pred=(torch.sigmoid(pred)>=0.5).float()
   target=target.float()
   true_positives = (pred * target).sum()
   predicted_positives = pred.sum()
   actual_positives = target.sum()
   precision = true_positives / (predicted_positives + 1e-8)  
   recall = true_positives / (actual_positives + 1e-8)
   return [precision,recall]
   

stopping = EarlyStopping(patience=5, delta=0.001)

def creating_model(train_loader,val_loader):
    i=0
    j=0
    for epoch in range(epochs):
      # Train
      train_precision, train_recall = 0, 0
      val_precision, val_recall = 0, 0
      for img, mask in train_loader:
        # Forward pass
        i=i+1
        print(f"i: {i}")
        output = model(img)
        
        a = cal_precision_recall(output, mask)
    
        train_precision=a[0]
        train_recall = a[1]
    
        # Calculate loss
        loss = criterion(output, mask)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      # Evaluate on validation data
      with torch.no_grad():
        val_loss = 0
        
        iou_score=0
        dice_score=0
        for img, mask in val_loader:
          j=j+1
          print(f"j: {j}")
          output = model(img)
          val_loss += criterion(output, mask)
          a=iou_val_dice_val(output,mask)
          iou_score+=a[0]
          dice_score+=a[1]
      
          a = cal_precision_recall(output, mask)
          val_precision += a[0]
          val_recall += a[1]

        avg_val_loss = val_loss / len(val_loader)
        avg_iou = iou_score / len(val_loader)
        avg_dice = dice_score / len(val_loader)
    
    
        train_precision /= len(train_loader)
        train_recall /= len(train_loader)
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)

        # Print training and validation losses
        print(f"Epoch: {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")
        print(f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
    
        stopping(avg_val_loss,model)
        if stopping.early_stop:
           break

if os.path.exists("D:\\projects\\VS\\project\\simple projects\\simple cell segmentation\\simple cell segmentation\\cell_segmentation.pth") is not True:
    
    creating_model(train_loader,val_loader)
    if os.path.exists("D:\\projects\\VS\\project\\simple projects\\simple cell segmentation\\simple cell segmentation\\cell_segmentation.pth") is not True:
        torch.save(model.state_dict(), "cell_segmentation.pth")


trans=tr.Compose([
            
            tr.Resize((512, 512)),
            
            tr.Grayscale(),
            
            tr.ToTensor(),
           
            tr.ConvertImageDtype(torch.float),
          ])
img_path=train_path+"\\SIMCEPImages_A03_C10_F1_s01_w2.tif"
label_path=mtrain_path+"\\SIMCEPImages_A03_C10_F1_s01_w2.tif"
state_dict=torch.load("cell_segmentation.pth")
input_img=Image.open(img_path).convert('L')
input_tensor=trans(input_img)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
model=Model2()
# Ensure the model is in evaluation mode
model.load_state_dict(state_dict)
model.eval()

# Predict
with torch.no_grad():
    output = model(input_batch)

# example, if output is a tensor that can be converted to an image
output_probs=torch.sigmoid(output)
output = (output_probs > 0.5).float()
output_image =tr.ToPILImage()(output.squeeze())






image1 = Image.open(img_path)
image2 = Image.open(label_path)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))


# Display the first image in the first subplot
axes[0].imshow(image1)
axes[0].set_title('original image')
axes[0].axis('off')

# Display the second image in the second subplot
axes[1].imshow(image2)
axes[1].set_title('ground truth mask')
axes[1].axis('off')

# Display the third image in the third subplot
axes[2].imshow(output_image)
axes[2].set_title('predicted mask')
axes[2].axis('off')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
