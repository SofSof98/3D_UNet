import numpy as np
import pandas as pd
import torch
import utils.Utils as utils
import utils.metrics as metrics



class Training_Model():

     def __init__(self, opt, model, train=True):
          self.isTrain = train
          self.opt = opt
          if self.isTrain:
               self.optimizer = utils.set_optimizer(opt, model_params=model.parameters())
               if opt.scheduling_lr is not None:
                     self.scheduler = utils.get_scheduler(self.optimizer, self.opt)

          

          # Enabling GPU
          self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
          self.model = model.to(self.device)

          self.loss_fn = utils.set_loss(self.opt,self.device)
          
     def forward(self, data):
          data = data.to(self.device)
          self.y_pred = self.model(data.unsqueeze(0))
          
     def backward(self, target):
          target = target.to(self.device)
          if self.opt.loss in ["dice","dice", "dice_bce", "iou","tversky", "focal_tv"]:
               self.loss = self.loss_fn(self.y_pred.squeeze(), target.float())
          else:
               self.loss = self.loss_fn(self.y_pred.squeeze(0), target)
          self.loss.backward()
          #self.loss_norm = self.loss/self.opt.batch_size
          #self.loss_norm.backward()
          
          
     def optimize_parameters(self, data, target):
          self.model.train()
          self.forward(data)

          self.optimizer.zero_grad()
          self.backward(target)
          self.optimizer.step()

     def optimize_parameters_accumulate_grd(self,data, target, iteration,idx_train):
          # if iteration % batch_size == 0 : optimizer.zero_grad()
          if iteration == 0: self.optimizer.zero_grad()
      
          self.model.train()
          self.forward(data)
          target = target.to(self.device)
          
          if self.opt.loss in ["dice","dice", "dice_bce", "iou","tversky", "focal_tv"]:
               self.loss = self.loss_fn(self.y_pred.squeeze(), target.float())
          else:
               self.loss = self.loss_fn(self.y_pred.squeeze(0), target)

          #if  (len(idx_train) - iteration) < len(idx_train)%self.opt.batch_size and len(idx_train)%self.opt.batch_size!=0 :
          if  (len(idx_train) - iteration) < len(idx_train)%self.opt.batch_size:
               self.opt.batch_size = self.opt.batch_size - len(idx_train)%self.opt.batch_size
               print('new:',self.opt.batch_size)
               
          self.loss_norm = self.loss/self.opt.batch_size
          self.loss_norm.backward()
          

          if (iteration) % self.opt.batch_size == 0 or iteration == len(idx_train):
               self.optimizer.step()
               self.optimizer.zero_grad()

     '''
     def optimize_parameters_accumulate_grd(self,data, target, iteration,idx_train):
          # if iteration % batch_size == 0 : optimizer.zero_grad()
          if iteration == 0: self.optimizer.zero_grad()
      
          self.model.train()
          self.forward(data)
 mmom 0,  self.backward(target)


          if (iteration) % self.opt.batch_size == 0 or iteration == len(idx_train):
               self.optimizer.step()
               self.optimizer.zero_grad() '''


     def validate(self, data, target):
          target = target.to(self.device)
          self.model.eval()
          self.forward(data)
        

          if self.opt.loss in ["dice","dice", "dice_bce", "iou","tversky", "focal_tv"]:
               self.loss = self.loss_fn(self.y_pred.squeeze(), target.float())
               #self.val_loss_2 = torch.nn.BCELoss()
               #CEL.append(val_loss_2(y_pred.squeeze(0), lesion).item())
          else:
                 self.loss = loss_fn(y_pred.squeeze(0), target)
    
     def get_prediction(self, target):
          pred = torch.where(self.y_pred.squeeze() > 0.5, 1, 0)
          return pred
       
     
     def get_score (self, target):
          target = target.to(self.device)
          pred = torch.where(self.y_pred.squeeze() > 0.5, 1, 0)
          return metrics.dice_coeff(pred.squeeze(), target.float()).cpu().detach().numpy()
    
     def get_iou_score (self, target):
          target = target.to(self.device)
          pred = torch.where(self.y_pred.squeeze() > 0.5, 1, 0)
          return metrics.IoU_coeff(pred.squeeze(), target.float()).cpu().detach().numpy()
 
     def get_loss(self):
          return self.loss.item()
     
     def print_stats(self, epoch, dice, idx, iteration):
          print("epoch: {0}, loss: {1}, dice: {2}, img idx: {3}, iterations: {4}".format(
                epoch, self.loss, dice, idx, iteration))


     def print_val_stats(self, epoch, dice, iou):
          print("epoch: {}, loss: {}, DSC: {}, Iou: {}".format(epoch, self.loss, dice, iou))

    
     def print_test_stats(self, idx, p_id, dice, iou):
          print("img idx: {}, p_id: {}, loss: {}, dice coeff.: {}, IoU coeff.: {}".format(idx, p_id, self.loss, dice, iou))
