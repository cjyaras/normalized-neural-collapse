import torch, os, json, pickle
import numpy as np
import scipy.linalg
from models.resnet import ResNet18, ResNet50
from models.resnet_vw import ResNet18VariableWidth
from tqdm import tqdm

class GeneralizedModel:

   def __init__(self, args, save_args=True):

      if args['device'] == 'cpu':
         print('Using CPU.')
         self.device = torch.device('cpu')
      else:
         if torch.cuda.is_available():
            print('GPU detected, using GPU.')
            self.device = torch.device('cuda')
         else:
            print('GPU not detected, using CPU.')
            self.device = torch.device('cpu')

      if args['model_type'] == 'ResNet18':
         print(f'Initializing Vanilla ResNet18')
         self.model = ResNet18(
            normalize=args['normalize'],
            num_classes=args['num_classes'],
            tau=args['tau']
         ).to(self.device)
      elif args['model_type'] == 'ResNet50':
         print(f'Initializing Vanilla ResNet50')
         self.model = ResNet50(
            normalize=args['normalize'],
            num_classes=args['num_classes'],
            tau=args['tau']
         ).to(self.device)
      elif args['model_type'] == 'ResNet18VariableWidth':
         print(f'Initializing ResNet18 with width:', args['width'])
         self.model = ResNet18VariableWidth(
            width=args['width'],
            normalize=args['normalize'],
            num_classes=args['num_classes'],
            tau=args['tau']
         ).to(self.device)
      else:
         raise ValueError('Model not implemented.')
      
      self.num_classes = args['num_classes']
      self.lr = args['lr']
      self.epochs = args['epochs']
      self.save_epoch = args['save_epoch']
      self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
      self.optimizer = torch.optim.SGD(
         self.model.parameters(), 
         lr=args['lr'], 
         momentum=args['momentum']
      )
      self.normalize = args['normalize']
      self.weight_decay = args['weight_decay']

      self.scheduler = torch.optim.lr_scheduler.StepLR(
         self.optimizer, 
         step_size=args['lr_decay_period'],
         gamma=args['lr_decay']
      )

      self.save_dir = args['save_dir']
      self.experiment_name = args['experiment_name']
      self.save_path = os.path.join(self.save_dir, self.experiment_name)

      if save_args:
         os.mkdir(self.save_path)

         with open(os.path.join(self.save_path, 'config.json'), 'w') as f:
            json.dump(args, f)

   def train(self, train_dataloader, test_dataloader=None, compute_nc=True):
      # Compute initial metrics
      self.model.eval()
      metrics = {
         'loss': [],
         'acc': [],
         'nc1': [],
         'nc2': [],
         'nc3': []
      }
      if compute_nc:
         loss, acc, nc1, nc2, nc3 = self.compute_loss_acc_nc(train_dataloader)
         metrics['loss'].append(loss)
         metrics['acc'].append(acc)
         metrics['nc1'].append(nc1)
         metrics['nc2'].append(nc2)
         metrics['nc3'].append(nc3)
      else:
         loss, acc = self.compute_loss_acc(train_dataloader)
         metrics['loss'].append(loss)
         metrics['acc'].append(acc)

      if test_dataloader is not None:
         metrics['test_loss'] = []
         metrics['test_acc'] = []
         metrics['test_nc1'] = []
         metrics['test_nc2'] = []
         metrics['test_nc3'] = []
         if compute_nc:
            test_loss, test_acc, test_nc1, test_nc2, test_nc3 = self.compute_loss_acc_nc(test_dataloader)
            metrics['test_loss'].append(test_loss)
            metrics['test_acc'].append(test_acc)
            metrics['test_nc1'].append(test_nc1)
            metrics['test_nc2'].append(test_nc2)
            metrics['test_nc3'].append(test_nc3)
         else:
            test_loss, test_acc = self.compute_loss_acc(test_dataloader)
            metrics['test_loss'].append(test_loss)
            metrics['test_acc'].append(test_acc)

      print('Before training')
      if compute_nc:
         print(f'Loss: {loss:0.2e} | Acc: {100*acc:0.2f}% | NC1: {nc1:0.2e} | NC2: {nc2:0.2e} | NC3: {nc3:0.2e}')
      else:
         print(f'Loss: {loss:0.2e} | Acc: {100*acc:0.2f}%')
      
      if test_dataloader is not None:
         print(f'Test Acc: {100*test_acc:0.2f}%')

      torch.save(self.model.state_dict(), os.path.join(self.save_path, 'epoch_0.pth'))

      for epoch in range(self.epochs):
         print(f'Epoch {epoch+1}/{self.n_epochs} | LR: {self.scheduler.get_last_lr()[-1]:0.2e}')
         self.model.train()
         for _, (inputs, targets) in enumerate(tqdm(train_dataloader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs, features = self.model(inputs)
            if self.normalize:
               loss = self.criterion(outputs, targets)
            else:
               W = self.model.fc.weight
               loss = self.criterion(outputs, targets) + \
                  self.weight_decay * torch.sum(W**2) / 2 + self.weight_decay * torch.sum(features**2) / 2
            self.step(loss)

         self.model.eval()
         if (epoch + 1) % self.save_epoch == 0:
            if compute_nc:
               loss, acc, nc1, nc2, nc3 = self.compute_loss_acc_nc(train_dataloader)
               metrics['loss'].append(loss)
               metrics['acc'].append(acc)
               metrics['nc1'].append(nc1)
               metrics['nc2'].append(nc2)
               metrics['nc3'].append(nc3)
               print(f'Loss: {loss:0.2e} | Acc: {100*acc:0.2f}% | NC1: {nc1:0.2e} | NC2: {nc2:0.2e} | NC3: {nc3:0.2e}')
            else:
               loss, acc = self.compute_loss_acc(train_dataloader)
               metrics['loss'].append(loss)
               metrics['acc'].append(acc)
               print(f'Loss: {loss:0.2e} | Acc: {100*acc:0.2f}%')

            if test_dataloader is not None:
               if compute_nc:
                  test_loss, test_acc, test_nc1, test_nc2, test_nc3 = self.compute_loss_acc_nc(test_dataloader)
                  metrics['test_loss'].append(test_loss)
                  metrics['test_acc'].append(test_acc)
                  metrics['test_nc1'].append(test_nc1)
                  metrics['test_nc2'].append(test_nc2)
                  metrics['test_nc3'].append(test_nc3)
               else:
                  test_loss, test_acc = self.compute_loss_acc(test_dataloader)
                  metrics['test_loss'].append(test_loss)
                  metrics['test_acc'].append(test_acc)

               print(f'Test Acc: {100*test_acc:0.2f}%')

            torch.save(self.model.state_dict(), os.path.join(self.save_path, f'epoch_{epoch+1}.pth'))
            
         self.scheduler.step()

      print('Saving metrics')
      with open(os.path.join(self.save_path, 'metrics.pkl'), 'wb') as f:
         pickle.dump(metrics, f)

   def step(self, loss):
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      self.model.normalize_fc()

   @torch.no_grad()
   def compute_loss_acc(self, dataloader, k=1):
      N = 0
      N_correct = 0
      total_loss = 0

      for inputs, targets in dataloader:
         inputs, targets = inputs.to(self.device), targets.to(self.device)
         outputs, _ = self.model(inputs)

         N_batch = targets.size(0)
         pred = outputs.topk(k, 1, True, True)[1].t()
         correct = pred.eq(targets.view(1, -1).expand_as(pred))
         N_correct += correct.reshape(-1).float().sum(0).item()

         batch_loss = self.criterion(outputs, targets)
         total_loss += batch_loss.item()
         N += N_batch
      
      acc = N_correct / N
      loss = total_loss / N

      return loss, acc

   @torch.no_grad()
   def compute_loss_acc_nc(self, dataloader):
      # Compute loss, accuracy, global mean and class means
      N = 0
      N_class = torch.zeros(self.num_classes, device=self.device)

      for inputs, targets in dataloader:
         inputs, targets = inputs.to(self.device), targets.to(self.device)
         outputs, features = self.model(inputs)

         N_batch = targets.size(0)
         N_class_batch = torch.tensor(
            [torch.sum(targets==k).item() for k in range(self.num_classes)],
            device=self.device
         )
         
         hG_batch_sum = torch.sum(features, dim=0)
         hbar_batch_sum = torch.cat(
            [torch.sum(features[targets==k], dim=0).unsqueeze(1) for k in range(self.num_classes)],
            dim=1
         )
         acc_batch_sum = torch.sum(torch.argmax(outputs, dim=1)==targets).item()
         loss_avg = self.criterion(outputs, targets).item()

         if N == 0:
            hG = hG_batch_sum / N_batch
            hbar = hbar_batch_sum / torch.clamp(N_class_batch, 1)
            acc = acc_batch_sum / N_batch
            loss = loss_avg
         else:
            hG = (N / (N + N_batch)) * hG + (1 / (N + N_batch)) * hG_batch_sum
            hbar = (N_class / torch.clamp(N_class + N_class_batch, 1)) * hbar + \
               (1 / torch.clamp(N_class + N_class_batch, 1)) * hbar_batch_sum
            acc = (N / (N + N_batch)) * acc + (1 / (N + N_batch)) * acc_batch_sum
            loss = (N / (N + N_batch)) * loss + (N_batch / (N + N_batch)) * loss_avg

         N += N_batch
         N_class += N_class_batch

      # Compute within class covariance
      N = 0

      for inputs, targets in dataloader:
         inputs, targets = inputs.to(self.device), targets.to(self.device)
         outputs, features = self.model(inputs)

         N_batch = targets.size(0)

         h_mean_diff = features.T - hbar[:, targets]
         Sigma_W_batch_sum = (h_mean_diff @ h_mean_diff.T)

         if N == 0:
            Sigma_W = Sigma_W_batch_sum / N_batch
         else:
            Sigma_W = (N / (N + N_batch)) * Sigma_W + (1 / (N + N_batch)) * Sigma_W_batch_sum

         N += N_batch

      # Compute between class covariance
      hbar_mean_diff = hbar - hG.unsqueeze(1)
      Sigma_B = (hbar_mean_diff @ hbar_mean_diff.T) / self.num_classes

      K = self.num_classes
      Sigma_W = Sigma_W.cpu().numpy()
      Sigma_B = Sigma_B.cpu().numpy()

      W = self.model.fc.weight.data.cpu().numpy().T
      H_bar = hbar.cpu().numpy()

      nc1 = 1 / K * np.trace(Sigma_W @ scipy.linalg.pinv(Sigma_B))
   
      # NC2
      gram = W.T @ W
      nc2 = np.linalg.norm(gram / np.linalg.norm(gram) - np.sqrt(1 / (K-1))*(np.eye(K) - 1/K))
      # nc2 = np.linalg.norm(W.T @ W - (K / (K-1) * (np.eye(K) - (1 / K))))
      
      # NC3
      dual = W.T @ H_bar
      nc3 = np.linalg.norm(dual / np.linalg.norm(dual) - np.sqrt(1 / (K-1))*(np.eye(K) - 1/K))
      # nc3 = np.linalg.norm(W.T @ H_bar - (K / (K-1) * (np.eye(K) - (1 / K))))

      return loss, acc, nc1, nc2, nc3
      