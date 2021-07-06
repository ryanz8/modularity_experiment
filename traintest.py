import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import time
import visdom
from . import config
from .util import log, getdev, accuracy
from .models import get_layers, get_weights

import wandb

import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

PATH = config.ROOT_DIR + '/models'

test_metrics = [
    nn.CrossEntropyLoss(reduction = 'sum'),
    accuracy(reduction = 'sum')]

def train(train_model, dataloader, params, lr=None, val_loader = None, dev = None, log_stats = True, save_checkpoints = True):

    if dev is None:
        dev = getdev()
    vis = config.vis
    n_epochs = int(params['n_steps']/len(dataloader))
    loss_history = []

    wandb.watch(train_model, log="all", log_freq = 100)

    criterion = nn.CrossEntropyLoss(reduction = 'sum')
    optimizer = optim.Adam(train_model.parameters(), lr=params['lr'], weight_decay = 0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = params['max_lr'], steps_per_epoch = len(dataloader), epochs = n_epochs)

    log('Starting training', vis, win='training')
    log(str(params), vis, win='training')
    start = time.time()

    loss_train_running_total = 0
    count_train = 0
    step_num = 0
    train_model.train()

    for epoch in range(n_epochs):
        for batch_num, (batch_in, batch_target) in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            batch_in, batch_target = batch_in.to(dev), batch_target.to(dev)

            # gradient step
            optimizer.zero_grad()
            output = train_model(batch_in)
            loss = criterion(output, batch_target)
            loss.backward()        
            optimizer.step()
            scheduler.step()

            # keep track of loss for display (not used for backprop)
            loss_train_running_total += loss
            count_train += np.prod(batch_target.shape)

            # checkpoint, validate and print stats
            if step_num % 25 == 0:
                loss_train = (loss_train_running_total/count_train).item()
                # we reset here since these are just used for logging, not backprop
                loss_train_running_total = 0
                count_train = 0

                if save_checkpoints:
                    torch.save(train_model.state_dict(), PATH+f'/step{step_num}.pth')

                # do validation if available
                loss_val = 0
                if val_loader is not None:
                    loss_val = evaluate(train_model, val_loader, test_metrics)
                    loss_val = loss_val.flatten()
                    train_model.train() # turn dropout back on

                loss_history.append((step_num, loss_train, loss_val[0], loss_val[1]))

                if log_stats:
                    log(f'Step {step_num:5d}, Epoch {epoch:3d}, Batch {batch_num:5d}, time: {time.time() - start:.2f} - train loss: {loss_train:.3f}', vis, win='training')
                    vis.line(X=np.array([step_num]), Y=np.array([loss_train]), win='loss', name='train_loss', update = 'append', opts={'showlegend':True, 'title':'Loss'})
                    vis.line(X=np.array([step_num]), Y=np.array([optimizer.param_groups[0]["lr"]]), win='lr', name='lr', update = 'append', opts={'showlegend':True, 'title':'Learning Rate'})

                    wandb.log({"train_loss": loss_train, "lr": optimizer.param_groups[0]["lr"]})

                    if val_loader is not None:
                        log(f'Step {step_num:5d}, Epoch {epoch:3d}, Batch {batch_num:5d}, time: {time.time() - start:.2f} - val loss: {loss_val[0]:.3f}, val acc: {loss_val[1]:.3f}', vis, win='training')
                        vis.line(X=np.array([step_num]), Y=np.array([loss_val[0]]), win='loss', name='val_loss', update = 'append', opts={'showlegend':True, 'title':'Loss'})
                        vis.line(X=np.array([step_num]), Y=np.array([loss_val[1]]), win='acc', name='val_acc', update = 'append', opts={'showlegend':True, 'title':'Accuracy'})                        
                        wandb.log({"val_loss": loss_val[0], "val_acc": loss_val[1]})

            step_num += 1

    return np.array(loss_history)

# Testing functions

def evaluate(evaluation_models, dataloader, metrics, dev = None):
    if dev is None:
        dev = getdev()
    if not isinstance(evaluation_models, list):
        evaluation_models = [evaluation_models]
    for m in evaluation_models:
        m.eval()

    loss_eval = np.stack([np.zeros_like(metrics, dtype = np.float) for m in evaluation_models])
    count_eval = np.stack([np.zeros_like(metrics, dtype = np.float) for m in evaluation_models])

    with torch.no_grad():
        # loop through batches first, since I think it takes longer to create batches
        for batch_num, (batch_in, batch_target) in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            batch_in, batch_target = batch_in.to(dev), batch_target.to(dev)

            for model_num, model in enumerate(evaluation_models, 0):
                output = model(batch_in)

                for i, m in enumerate(metrics):
                    loss_eval[model_num, i] += m(output, batch_target)
                    count_eval[model_num, i] += np.prod(batch_target.shape)

    loss_eval = loss_eval/count_eval
    return loss_eval

def test(testing_models, test_loader, dev = None):
    if dev is None:
        dev = getdev()
    vis = config.vis

    if not isinstance(testing_models, list):
        evaluation_models = [testing_models]
    
    loss_test = evaluate(evaluation_models, test_loader, test_metrics)

    for model_loss in loss_test:
        # print("Test set results:",
        #       *[repr(m) + ": {:.6f} ".format(x.item()) for x, m in zip(model_loss, test_metrics)])
        log('Test set results: ' + ' '.join([repr(m) + ": {:.6f} ".format(x.item()) for x, m in zip(model_loss, test_metrics)]), vis, win='training')
    return loss_test



def train_to(emb_model, ref_model, params, fixed_embeddings = None, fixed_layer_nums = None):
    '''
    input_embedding: the input embedding to use. Layer will be locked (not trained)
    '''
    vis = config.vis

    # criterion = nn.MSELoss(reduction = 'mean')
    n_steps = params['n_steps_ref']
    log_stats = True

    optimizer = optim.Adam(emb_model.parameters(), lr=params['lr_ref'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience=100)
    
    # freeze reference just in case
    for param in ref_model.parameters():
        param.requires_grad = False

    ref_weights = [dict(l.named_parameters())['weight'].T for l in ref_model.linears]
    ref_biases = [dict(l.named_parameters())['bias'] for l in ref_model.linears]

    # copy all the data that we don't need to train
    sd = emb_model.state_dict()
    for i, bias in enumerate(ref_biases):
        sd[f'biases.{i}'] = bias
    # copy input embedding if asked for
    if fixed_embeddings is not None:
        for fixed_layer_num, fixed_embedding in zip(fixed_layer_nums, fixed_embeddings):
            sd[f'embs.{fixed_layer_num}.weight']= fixed_embedding
        
    # copy to the embedding model, these will not be touched
    emb_model.load_state_dict(sd)

    # lock all the ones we copied
    for bias in emb_model.biases:
        bias.requires_grad = False
    if fixed_embeddings is not None:
        for fixed_layer_num in fixed_layer_nums:
            emb_model.embs[fixed_layer_num].weight.requires_grad = False



    log('Starting training', vis, win='training')
    start = time.time()
    loss_history = []
    
    for step_num in range(n_steps):
        # zero the parameter gradients
        optimizer.zero_grad()

        emb_weights = get_weights(emb_model.embs)

        # If we're using weight biases and weight scaling

    #     for i in range(len(train_model.weight_biases)):
    #     weights[i] = weights[i] * train_model.weight_scaling[i] + train_model.weight_biases[i]

        loss = 0
        for ref_w, emb_w, in zip(ref_weights, emb_weights):
            loss += torch.norm(ref_w - emb_w, p="fro")**2/torch.numel(ref_w)

            #criterion(ref_w, emb_w)


        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # print statistics
        if log_stats and step_num % 50 == 0:
            loss_train = loss.item()
            log(f'Step {step_num:3d}, time {time.time() - start:.2f}- loss: {loss_train:.5f}', vis, win='training')
            loss_history.append(loss_train)
            vis.line(X=np.array([step_num]), Y=np.array([loss_train]), win='reference loss', name='train_loss', update = 'append', opts={'title':'Train to Reference MSE Loss'})
