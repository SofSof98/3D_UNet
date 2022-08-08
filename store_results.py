# code to fill csv file for each model trained
import os
import  csv

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created folder(s) {}".format(path))
    else:
        print("Folder(s) {} already exist(s).".format(path))


path = "stored_results"
new = True
if new:
    create_folder(path)
    with open(path + "/results.csv", "w", newline="") as file:
                writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["model", "training epochs", "loss function", "optimizer", "lr", "lr scheduling", "normalization","augmentation", "filter scale", "batch_size", "val loss", \
                                "epoch", "val dice", "epoch", "val iou", "epoch", "test ce loss", "test dice", "test iou" ])


model = input("Model trained:\n")
tr_epoch = input("Number of training epochs:\n")
loss_fn = input("Loss function used:\n")
opt = input("Optimizer used:\n")
lr = input("lr used:\n")
lr_s = input("lr scheduling?\n")
norm = input("Normalization type:\n")
aug = input("Did you perform augmentation?\n")
f_s = input("Filter scale used:\n")
batch = input("Batch size used:\n")
val_loss = input("Best validation loss obtained:\n")
epoch_l = input("Corresponding epoch:\n")
val_dice = input("Best validation dice obtained:\n")
epoch_d = input("Corresponding epoch:\n")
val_iou = input("Best validation IoU obtained:\n")
epoch_i = input("Corresponding epoch:\n")
test_loss = input("Test ce loss:\n")
test_dice = input("Test Dice:\n")
test_iou = input("Test IoU:\n")


with open(path + "/results.csv", "a", newline="") as file:
                    writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([model, tr_epoch, loss_fn, opt, lr, lr_s, norm, aug, f_s, batch, val_loss, epoch_l, val_dice, epoch_d,\
                                     val_iou, epoch_i, test_loss, test_dice, test_iou])
                

print("Model Information entered and stored")
