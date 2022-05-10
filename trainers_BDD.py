# -*- coding: utf-8 -*-
"""
This files's functions are almost written by SENN's authors to train SENN.
We modified so as to fit the semi-supervised fashion.
"""

# standard imports
import sys
import os
import tqdm
import time
import pdb
import shutil
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Local imports
from SENN.utils import AverageMeter

#===============================================================================
#====================      REGULARIZER UTILITIES    ============================
#===============================================================================
"""
def compute jacobian:
Inputs: 
    x: encoder's output
    fx: prediction
    device: GPU or CPU
Return:
    J: Jacobian
NOTE: This function is not modified from original SENN
"""
def compute_jacobian(x, fx, device):
    # Ideas from https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059/2
    b = x.size(0)
    n = x.size(-1)
    m = fx.size(-1)
    
    J = []
    for i in range(m):
        #print(i)
        grad = torch.zeros(b, m)
        grad[:,i] = 1
        if x.is_cuda:
            grad  = grad.cuda(device)
        g = torch.autograd.grad(outputs=fx, inputs = x, grad_outputs = grad, create_graph=True, only_inputs=True)[0]
        J.append(g.view(x.size(0),-1).unsqueeze(-1))
    J = torch.cat(J,2)
    return J


#===============================================================================
#==================================   TRAINERS    ==============================
#===============================================================================
"""
def save_checkpoint: 
    save best model
Inputs: 
    state: several values in the current status
    is_best: flag whether the best model or not
    outpath: save path
Return:
    None
NOTE: This function is not modified from original SENN
"""
def save_checkpoint(state, is_best, outpath):
    if outpath == None:
        outpath = os.path.join(script_dir, 'checkpoints')

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filename = os.path.join(outpath, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(outpath,'model_best.pth.tar'))


"""
Class ClassificationTrainer:
    Executes train, val and test and includes many functions... 
Here is an overview of each function.
def train: 
    iterate function until last epoch, called from main function in main_cub.py, like trainer.train(...)
def train_batch: 
    only print error message
def concept_learning_loss_for_weak_supervision: 
    compute losses of known concepts and discriminator (added by Sawada)
def train_epoch: 
    train 1 epoch, called from train function
def validate: 
    validate after end of each epoch, called from train function
def test_and_save: 
    after training, this function tests by test data and save the results
def concept_error: 
    compute known concept error, called from train_epoch, validate, and test_and_save functions
def binary_accuracy: 
    compute binary accuracy of task (not use in our case, is not modified by Sawada)
def accuracy: 
    compute accyracy of task (is not modified by Sawada)
def plot_losses: 
    make figures of losses (is not modified by Sawada)
"""
class ClassificationTrainer():
    
    """
    def train: 
        iterate function until last epoch, called from main function in main_cub.py, like trainer.train(...)
    Inputs:
        train_loader: training data
        val_loader: validation data
        epochs: # of epochs for training
        save_path: save file's path
    Returns:
        None
    """    
    def train(self, train_loader, val_loader = None, epochs = 10, save_path = None):
        best_prec1 = 0
        for epoch in range(epochs):
            
            # go to train_epoch function
            self.train_epoch(epoch, train_loader)

            # validate evaluation
            if val_loader is not None:
                val_prec1  =1
                val_prec1 = self.validate(val_loader)

            # remember best prec@1 and save checkpoint
            is_best = val_prec1 > best_prec1
            best_prec1 = max(val_prec1, best_prec1)
            if save_path is not None:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'lr': self.args.lr,
                    'theta_reg_lambda': self.args.theta_reg_lambda,
                    'theta_reg_type': self.args.theta_reg_type,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : self.optimizer.state_dict(),
                    'pretrained' : self.pretrained_model.state_dict(),
                    'model': self.model  
                 }, is_best, save_path)

        # end message
        print('Training done')

    """
    def train_batch: 
        only print error message
    Inputs:
        None    
    Returns:
        Error Message
    """    
    def train_batch(self):
        raise NotImplemented('ClassificationTrainers must define their train_batch method!')        

    """
    def concept_learning_loss_for_weak_supervision: (added by Sawada)
        compute losses of known concepts and discriminator
    Inputs:
        inputs: output of Faster RCNN
        all_losses: loss file (saving all losses to print)
        concepts: correct concepts
        cbm: flag whether use CBM's model or not
        epoch: the number of current epoch. Current version does not use. But if you want to make process using epoch, please use it.
    Returns:
        info_loss: loss weighed adding discrminator's loss and known concept loss
        hh_labeled_list: predicted known concepts        
    """    
    def concept_learning_loss_for_weak_supervision(self, inputs, all_losses, concepts, cbm, senn, epoch):

        # compute predicted known concepts by inputs
        # real uses the discriminator's loss
        hh_labeled_list, h_x, real = self.model.conceptizer(inputs)

        
        if not senn:

            # compute losses of known concepts            
            labeled_loss = F.binary_cross_entropy(hh_labeled_list[0], concepts[0].to(self.device))
            for i in range(1,len(hh_labeled_list)):
                labeled_loss = labeled_loss + F.binary_cross_entropy(hh_labeled_list[i], concepts[i].to(self.device))

            #MSE loss version for known concepts
            #labeled_loss = F.mse_loss(hh_labeled_list,concepts)
            #labeled_loss = labeled_loss*len(concepts[0])        
    
        
        """
        compute discriminator's loss
        Discriminator's architecture is inspired by the global DeepInfoMax model.
        DeepInfoMax@ICLR2019: https://arxiv.org/pdf/1808.06670.pdf
        """
        if not cbm:            

            Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

            labs_real = Variable(Tensor(inputs.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
            labs_fake = Variable(Tensor(inputs.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)

            # main loss
            
            # select random index
            rand_index = torch.randperm(h_x.size()[0]).cuda(self.device)

            # compute fake data (concate[rand_inputs, known_conepts, unknown_concepts]) and output of the discriminator
            fake = self.model.conceptizer.decode(inputs[rand_index], hh_labeled_list, h_x)
            
            """
            compute loss
            if you use DeepInfoMax loss, please remove the comment outs
            """
            info_loss = self.gamma*(F.mse_loss(real, labs_real) + F.mse_loss(fake, labs_fake))

            # save loss (only value) to the all_losses list
            all_losses['info'] = info_loss.data.cpu().numpy()
            

        if cbm: # Standard CBM does not use decoder
            info_loss = self.eta*labeled_loss
        elif senn:
            info_loss = info_loss
        else:
            info_loss += self.eta*labeled_loss
                
        if not senn:
            # save loss (only value) to the all_losses list
            all_losses['labeled_h'] = labeled_loss.data.cpu().numpy()

        # use in def train_batch (class GradPenaltyTrainer)
        return info_loss, hh_labeled_list
    
    """
    def train_epoch: 
        train 1 epoch, called from train function
    Inputs:
        epoch: the number of current epoch
        train_loader: training data
    Returns:
        None
    Outputs made in this function:
        print errors, losses of each epoch
    """        
    def train_epoch(self, epoch, train_loader):

        # initialization of print's values
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topc1 = AverageMeter()

        # switch to train mode
        self.model.train()
        self.pretrained_model.train()

        end = time.time()
        
        for i, (inputs, targets, concepts) in enumerate(train_loader, 0):

            # measure data loading time
            data_time.update(time.time() - end)
            
            # get the inputs
            if self.cuda:
                inputs = inputs.cuda(self.device)
                concepts = concepts.cuda(self.device)
                targets = targets.cuda(self.device)

            # go to def train_batch (class GradPenaltyTrainer)
            outputs, loss, loss_dict, hh_labeled, pretrained_out = self.train_batch(inputs, targets, concepts, epoch)
            
            # add to loss_history
            loss_dict['iter'] = i + (len(train_loader)*epoch)
            self.loss_history.append(loss_dict)

            # measure accuracy and record loss
            if self.nclasses > 4:
                # mainly use this line (current)
                prec1, _ = self.accuracy(outputs.data, targets.data, topk=(1, 5))
            elif self.nclasses in [3,4]:
                prec1, _ = self.accuracy(outputs.data, targets.data, topk=(1,self.nclasses))
            else:
                prec1, _ = self.binary_accuracy(outputs.data, targets.data), [100]

            # update each value of print's values
            losses.update(loss.data.cpu().numpy(), pretrained_out.size(0))
            top1.update(prec1[0], pretrained_out.size(0))
             
            if not self.args.senn:                
                # measure accuracy of concepts
                err = self.concept_error(hh_labeled.data, concepts)

                # update print's value
                topc1.update(err, pretrained_out.size(0))
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if not self.args.senn:
                # print values of i-th iteration in the current epoch
                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]  '
                          'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                          'Loss {loss.val:.4f} ({loss.avg:.4f})  '.format(
                           epoch, i, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses))
            else:
                # print values of i-th iteration in the current epoch
                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]  '
                          'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                          'Loss {loss.val:.4f} ({loss.avg:.4f})  '.format(
                           epoch, i, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses))

        # optimizer's schedule update based on epoch
        self.scheduler.step(epoch) 
        self.pre_scheduler.step(epoch) 


    """
    def validate: 
        validate after end of each epoch, called from train function
    Inputs:
        val_loader: validation data
    Returns:
        top1.avg: use whether models save or not
    Outputs made in this function:
        print errors, losses of each epoch
    NOTE: many code is the same to def train_epoch
    """        
    def validate(self, val_loader, fold = None):

        # initialization of print's values
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topc1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        self.pretrained_model.eval()

        end = time.time()
        for i, (inputs, targets, concepts) in enumerate(val_loader):

            # get the inputs
            if self.cuda:
                inputs, targets, concepts = inputs.cuda(self.device), targets.cuda(self.device), concepts.cuda(self.device)

            # compute output        
            pretrained_out = self.pretrained_model(inputs)
            output = self.model(pretrained_out)

            # prediction_criterion is defined in __init__ of "class GradPenaltyTrainer"
            loss = self.prediction_criterion(output, targets)

            # measure accuracy and record loss
            if self.nclasses > 4:
                # mainly use this line (current)
                prec1, _ = self.accuracy(output.data, targets, topk=(1, 5))
            elif self.nclasses == 3:
                prec1, _ = self.accuracy(output.data, targets, topk=(1,3))
            else:
                prec1, _ = self.binary_accuracy(output.data, targets), [100]

            # update each value of print's values
            losses.update(loss.data.cpu().numpy(), pretrained_out.size(0))
            top1.update(prec1[0], pretrained_out.size(0))

                        
            # measure accuracy of concepts
            hh_labeled, _, _ = self.model.conceptizer(pretrained_out)
            if not self.args.senn:
                err = self.concept_error(hh_labeled.data, concepts)

                # update print's value
                topc1.update(err, pretrained_out.size(0))
                        
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not self.args.senn:
                # print values of i-th iteration in the current epoch
                if i % self.print_freq == 0:
                    print('Val: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           i, len(val_loader), batch_time=batch_time, loss=losses))
            else:
                # print values of i-th iteration in the current epoch
                if i % self.print_freq == 0:
                    print('Val: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           i, len(val_loader), batch_time=batch_time, loss=losses))


        # top1.avg: use whether models save or not
        return top1.avg

    
    """
    def test_and_save: 
        after training, this function tests by test data and save the results
    Inputs:
        test_loader: test data
        save_file_name: file name to save the predicted and correct concepts, predicted and correct classes...
    Returns:
        None
    Outputs made in this function:
        print errors, losses of each epoch
    NOTE: many code is the same to def train_epoch
    """        
    def test_and_save(self, test_loader, save_file_name, fold = None):

        # initialization of print's values
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topc1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        self.pretrained_model.eval()

        end = time.time()

        # open the save file
        fp = open(save_file_name,'a')
        for i, (inputs, targets, concepts) in enumerate(test_loader):

            # get the inputs
            if self.cuda:
                inputs, targets, concepts = inputs.cuda(self.device), targets.cuda(self.device), concepts.cuda(self.device)

            # compute output
            pretrained_out = self.pretrained_model(inputs)
            output = self.model(pretrained_out)

                         
            # prediction_criterion is defined in __init__ of "class GradPenaltyTrainer"
            loss = self.prediction_criterion(output, targets)

            # measure accuracy and record loss
            if self.nclasses > 4:
                # mainly use this line (current)
                prec1, _ = self.accuracy(output.data, targets, topk=(1, 5))
            elif self.nclasses == 3:
                prec1, _ = self.accuracy(output.data, targets, topk=(1,3))
            else:
                prec1, _ = self.binary_accuracy(output.data, targets), [100]

            # update each value of print's values
            losses.update(loss.data.cpu().numpy(), pretrained_out.size(0))
            top1.update(prec1[0], inputs.size(0))

            # measure accuracy of concepts
            hh_labeled, hh, _ = self.model.conceptizer(pretrained_out)
            if not self.args.senn:
                err = self.concept_error(hh_labeled.data, concepts)

                # update print's value
                topc1.update(err, pretrained_out.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not self.args.senn:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    #pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_labels = hh_labeled.data.to("cpu")
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_labels = hh_labeled
                    concept_nolabels = hh
                    attr = concepts


                # save to the file
                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f,"%(targets[j][k]))
                        #fp.write("%d,%d,"%(targets[j][k],pred_labels[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f,"%(pred_labels[j][k]))
                    for k in range(concept_labels.shape[1]):
                        fp.write("%f,"%(concept_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f,"%(concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f,"%(attr[j][k]))
                    fp.write("\n")

                # print values of i-th iteration
                if i % self.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           i, len(test_loader), batch_time=batch_time, loss=losses))

            else:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    #pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_nolabels = hh
                    attr = concepts

                # save to the file
                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f,"%(targets[j][k]))
                        #fp.write("%d,%d,"%(targets[j][k],pred_labels[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f,"%(pred_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f,"%(concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f,"%(attr[j][k]))
                    fp.write("\n")

                
                # print values of i-th iteration
                if i % self.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           i, len(test_loader), batch_time=batch_time, loss=losses))


        # close the save_file_name
        fp.close()
    
    
    """
    def concept_error: (added by Sawada)
        compute known concept error, called from train_epoch, validate, and test_and_save functions
    Inputs:
        output: predicted concepts
        target: correct concepts
    Returns:
        err: concept's error
    NOTE: many code is the same to def binary_accuracy
    """        
    def concept_error(self, output, target):
        err = torch.Tensor(1).fill_((output.round().eq(target)).float().mean()*100)
        err = (100.0-err.data[0])/100
        return err
    
    """
    def binary_accuracy: 
        compute binary accuracy of task (not use in our case)
    Inputs:
        output: predicted task class
        target: correct task class
    Returns:
        err: task's error
    NOTE: This function is not modified by Sawada
    """        
    def binary_accuracy(self, output, target):
        """Computes the accuracy"""
        return torch.Tensor(1).fill_((output.round().eq(target)).float().mean()*100)

    """
    def accuracy:
        compute accuracy of task
    Inputs:
        output: predicted task class
        target: correct task class
        topk: compute top1 accuracy and topk accuracy (currently k=5)
    Returns:
        res: task's error
    NOTE: This function is not modified by Sawada
    """        
    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        #pred = pred.t()
        #correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct = pred.eq(target.long())
        
        # if topk = (1,5), then, k=1 and k=5
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    """
    def plot_losses: 
        make figures of losses
    Inputs:
        save_path: path of save file
    Returns:
        res: None
    NOTE: This function is not modified by Sawada
    """        
    def plot_losses(self, save_path = None):
        loss_types = [k for k in self.loss_history[0].keys() if k != 'iter']
        losses = {k: [] for k in loss_types}
        iters  = []
        for e in self.loss_history:
            iters.append(e['iter'])
            for k in loss_types:
                losses[k].append(e[k])
        fig, ax = plt.subplots(1,len(loss_types), figsize = (4*len(loss_types), 5))
        if len(loss_types) == 1:
            ax = [ax] # Hacky, fix
        for i, k in enumerate(loss_types):
            ax[i].plot(iters, losses[k])
            ax[i].set_title('Loss: {}'.format(k))
            ax[i].set_xlabel('Iters')
            ax[i].set_ylabel('Loss')
        if save_path is not None:
            plt.savefig(save_path + '/training_losses.pdf', bbox_inches = 'tight', format='pdf', dpi=300)
        #plt.show(block=False)

"""
class GradPenaltyTrainer: 
    Gradient Penalty Trainer. uses different penalty:
    || df/dx - dh/dx*theta  || (=  || dth/dx*h  || )
Here is overview of def functions
def __init__:
    initial setting
def train_batch:
    main training function of each iteration
def compute_conceptizer_jacobian:
    compute jacobian of output (not modified by Sawada)
def compute_conceptizer_jacobian_aux:
    compute jacobian of aux. output (added by Sawada)
"""        
class GradPenaltyTrainer(ClassificationTrainer):
    
    """
    def __init__:
        initial setting
        define self variable (e.g., self.gamma...)
    Inputs:
        model: (conceptizer,parametrizer,aggregator) for output of inception v.3
        model_aux: (conceptizer,parametrizer,aggregator) for aux. of inception v.3
        pretrained_model: inception v.3
        args: hyparparameters we set
        device: GPU or CPU
    Returns:
        None
    """
    def __init__(self, model, pretrained_model, args, device):
        
        # hyparparameters used in the loss function
        self.lambd = args.theta_reg_lambda if ('theta_reg_lambda' in args) else 1e-6 # for regularization strenght
        self.eta = args.h_labeled_param if ('h_labeled_param' in args) else 0.0 # for wealky supervised 
        self.gamma = args.info_hypara if ('info_hypara' in args) else 0.0 # for wealky supervised 

        
        # use the gradient norm conputation
        self.norm = 2

        # set models
        self.pretrained_model = pretrained_model
        self.model = model
        
        # others
        self.args = args
        self.cuda = args.cuda
        self.device = device

        self.nclasses = args.nclasses

        # select prediction_criterion for task classification 
        if args.nclasses <= 2 and args.objective == 'bce':
            self.prediction_criterion = F.binary_cross_entropy_with_logits
        elif args.nclasses <= 2:# THis will be default.  and args.objective == 'bce_logits':
            self.prediction_criterion = F.binary_cross_entropy # NOTE: This does not do sigmoid itslef
        elif args.objective == 'cross_entropy':
            self.prediction_criterion = F.cross_entropy
        elif args.objective == 'mse':
            self.prediction_criterion = F.mse_loss            
        else:
            #self.prediction_criterion = F.nll_loss # NOTE: To be used with output of log_softmax
            #BCE loss for multible labels  
            self.prediction_criterion = self.BCE_forloop
             
            
        self.learning_h = True

        # acumulate loss to make loss figure
        self.loss_history = []  
        
        # use to print error, loss
        self.print_freq = args.print_freq

        """
        select optimizer
        self.optimizer: [conceptizer, parametrizer, aggregator]
        self.aux_optimizer: [conceptizer, parametrizer, aggregator] for aux. output
        self.pre_optimizer: for pretrained model
        """
        if args.opt == 'adam':
            optim_betas = (0.9, 0.999)
            self.optimizer = optim.Adam(self.model.parameters(), lr= args.lr, betas=optim_betas)
            self.pre_optimizer = optim.Adam(self.pretrained_model.parameters(), lr= args.lr, betas=optim_betas)
        elif args.opt == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr = args.lr)
            self.pre_optimizer = optim.RMSprop(self.pretrained_model.parameters(), lr = args.lr)
        elif args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum=0.9)
            self.pre_optimizer = optim.SGD(self.pretrained_model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum=0.9)
            
        # set scheduler for learning rate
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        self.pre_scheduler = torch.optim.lr_scheduler.StepLR(self.pre_optimizer, step_size=20, gamma=0.1)
        
        
    """
    def train_batch:
        training function of each batch
    Inputs:
        inputs: samples of each batch
        targets: labels of each batch
        concepts: correct concepts of each batch
        epoch: the number of current epoch. Current version does not use. 
    Returns:
        pred: task predicted results
        loss: loss for training
        all_losses: all losses for print
        hh_labeled: predicted known concepts
        pretrained_out: output of encoder (pretrained model)
    """
    def train_batch(self, inputs, targets, concepts, epoch):

        # Init
        self.optimizer.zero_grad()
        self.pre_optimizer.zero_grad()

        inputs, targets, concepts = Variable(inputs), Variable(targets), Variable(concepts)
            

        # Predict
        pretrained_out = self.pretrained_model(inputs)
        pred = self.model(pretrained_out)


        # Calculate loss
        pred_loss = self.prediction_criterion(pred, targets)
        
        
        # save loss (only value) to the all_losses list
        all_losses = {'prediction': pred_loss.cpu().data.numpy()}
            
        # compute loss of known concets and discriminator
        h_loss, hh_labeled = self.concept_learning_loss_for_weak_supervision(pretrained_out, all_losses, concepts, self.args.cbm, self.args.senn, epoch)

        # to be simplify, redifine
        inputs = pretrained_out

        # following process is the constraint of parametrizer see paper of SENN
        # SENN: https://arxiv.org/abs/1806.07538
        # (V1)  || dh/dx*theta - df/dx  || =  || dth/dx*h  ||  (V2)
        # For V1:
        dF = compute_jacobian(inputs, pred, self.device)

        dH  = self.compute_conceptizer_jacobian(inputs)
        prod = torch.bmm(dH, self.model.thetas)

        ## For V2:
        grad_penalty = (prod - dF).norm(self.norm) 


        # save loss (only value) to the all_losses list
        all_losses['grad_penalty'] = grad_penalty.data.cpu().numpy()

        # total loss to train models
        loss = pred_loss + self.lambd*grad_penalty + h_loss

        # back propagation
        loss.backward()

        # update each model
        self.optimizer.step()
        self.pre_optimizer.step()

        return pred, loss, all_losses, hh_labeled, pretrained_out

    """
    def compute_conceptizer_jacobian:
        compute jacobian of output
    Inputs:
        x: output of encoder
    Returns:
        Jh: jacobian
    NOTE: This function is not modified by Sawada
    """
    def compute_conceptizer_jacobian(self, x):
        h = self.model.concepts
        Jh = compute_jacobian(x, h.squeeze(), self.device)
        assert list(Jh.size()) == [x.size(0), x.view(x.size(0),-1).size(1), h.size(1)]
        return Jh

    def BCE_forloop(self,tar,pred):
        
        loss = F.binary_cross_entropy(tar[0], pred[0])
        
        for i in range(1,len(tar)):
            loss = loss + F.binary_cross_entropy(tar[i], pred[i])
            
        return loss

                                                         
