import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from unet import UNet
import numpy as np
from dataset import hrgldd_dataset
from tqdm import tqdm
from loss import dice_loss
from dice_score import dice_score



if __name__ == "__main__":
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 16
    EPOCHS = 1
    DATA_PATH = ""
    MODEL_SAVE_PATH = ""


    path_to_testX = ''
    path_to_testY = ''
    path_to_trainX = ''
    path_to_trainY = ''
    path_to_valX = ''
    path_to_valY = ''

    data_testY = np.load(path_to_testY)
    data_testX = np.load(path_to_testX)
    data_trainX = np.load(path_to_trainX)
    data_trainY = np.load(path_to_trainY)
    data_valX = np.load(path_to_valX)
    data_valY = np.load(path_to_valY)

    train_dataset = hrgldd_dataset(data_trainX, data_trainY)
    val_dataset = hrgldd_dataset(data_valX, data_valY)
    test_dataset = hrgldd_dataset(data_testX, data_testY)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = BATCH_SIZE,
                                  shuffle = True)
    val_dataloader = DataLoader(dataset = val_dataset,
                                batch_size = BATCH_SIZE,
                                shuffle = True)

    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 3 , verbose = True)

    for epoch in tqdm(range(EPOCHS)):
        train_running_loss = 0
        model.train()
        for index, x_y in enumerate(tqdm(train_dataloader)):
            img = x_y[0].float().permute(0,3,1,2).to(device)
            mask = x_y[1].float().permute(0,3,1,2).to(device)

            optimizer.zero_grad()
            outputs = model(img) 
            loss = dice_loss(outputs, mask)
            loss.backward() #Find Gradients by Backward Pass
            optimizer.step() #Update the weights

            train_running_loss += loss.item()

        train_loss = train_running_loss / len(train_dataloader)

 # --------------- Validation Part ------------------------ #

        model.eval()
        val_running_loss = 0
        val_dice = 0
        with torch.no_grad():
            for index, x_y in enumerate(tqdm(val_dataloader)):
                img = x_y[0].float().permute(0,3,1,2).to(device)
                mask = x_y[1].float().permute(0,3,1,2).to(device)

                y_pred = model(img)
                loss = dice_loss(y_pred, mask)
                val_running_loss += loss.item()

                val_dice += dice_score(y_pred, mask).item()
            
            val_loss = val_running_loss / len(val_dataloader)
            val_dice = val_dice / len(val_dataloader)
        
        print("--" * 30)
        print(f"Train Loss EPOCH {epoch +1}: {train_loss:.4f}")
        print(f"Val Loss EPOCH {epoch +1} : {val_loss:.4f}")
        print(f"Val Dice Score EPOCH {epoch + 1} : {val_dice:.4f}")
        print(f"Current Learning Rate : {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model with val loss : {val_loss : .4f}")

        scheduler.step(val_loss)
