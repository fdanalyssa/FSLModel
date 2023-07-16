
import torch
import wandb
from tqdm.auto import tqdm
from utils import save_model as save_model
import utils


def train_loop(model, dataloader, loss_fn, optimizer, device):


  train_loss, train_acc = 0, 0 # Setup train loss and train accuracy values

  model.train() #Set model to train mode

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      
      # Send data to target device
      X, y = X.to(device), y.to(device)

      optimizer.zero_grad() #Clear Gradients 

      y_pred = model(X) #Compute Output

      
      loss = loss_fn(y_pred, y) # Calculate  and accumulate loss
      train_loss += loss.item() 


      loss.backward() #Loss backward

      optimizer.step() #Optimizer step

      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)
  
  
  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  
  return train_loss, train_acc



def test_loop(model, dataloader, loss_fn, device):
  
  # Put model in eval mode
  model.eval() 

  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0
  
  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):      

          # Send data to target device
          X, y = X.to(device), y.to(device)
          # 1. Forward pass
          test_pred_logits = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()
        
          # Calculate and accumulate accuracy
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch  
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc



def wandbinit(lr=0.0001, architecture = "CNN", dataset = "Custom FSL Dataset", batch_size = 64, epoch = 20, optimizer = "RMSProp"):
           # start a new wandb run to track this script
  wandb.init(
  # set the wandb project where this run will be logged
  project="FILOSIGN-Final-3",
        
  # track hyperparameters and run metadata
  config={
  "learning_rate": lr,
  "architecture": architecture,
  "dataset": dataset,
  "batchsize": batch_size,
  "epochs": epoch,
  "optimizer": optimizer
        }
    )


def train(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, epochs, device, lr, architecture, dataset, batch_size, opt_name, model_path, cfm_name, csv_name):
  
    
  wandbinit(lr, architecture, dataset, batch_size, epochs, opt_name)
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": [],
      "F1": [],
      "Precision": [],
      "Recall": []
  }
    

  best_test_acc = 0
  best_test_acc_epoch = 0
  test_acc = 0
  model_save_path = model_path
  patience = 15

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      

      train_loss, train_acc = train_loop(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,                                   
                                          optimizer=optimizer,
                                          device=device)
  
      test_loss, test_acc = test_loop(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

      f1, precision_macro, recall_macro, df = utils.Evaluation_Metrics(model=model,
                        dataloader=test_dataloader,
                        device=device)
      
      scheduler.step() #Scheduler step

      print(' Epoch {}, lr {}'.format(
        epoch + 1, optimizer.param_groups[0]['lr']))
      
      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f} | "
          f"f1: {f1:.4f} | "
          f"precision_macro: {precision_macro:.4f} | "
          f"recall_macro: {recall_macro:.4f}"
      )


    # Check if test accuracy is better than current best
      if test_acc > best_test_acc:
        # If so, save model to path
            torch.save(obj=model.state_dict(), f= model_save_path)

            # Update best test accuracy
            best_test_acc = test_acc

            #Update best test epoch number
            best_test_acc_epoch = epoch

            # Save the model state_dict()
            print(f"[INFO] Saving model to: {model_save_path} with test accuracy: {test_acc:.4f} with learning rate {lr:.4f}")

              #Creating a confusion matrix
            utils.confusion_matrix_plot(model, test_dataloader, device, cf_plot_name= cfm_name)

            #Creating a csv file
            print(f"[INFO] Saving Classification Report to : {csv_name}")
            df.to_csv(csv_name, index=False)
            print(f"[INFO] Saving Completed")
      elif (epoch - best_test_acc_epoch) > patience:
            print(f"[INFO] Early stopping since test accuracy has not improved for {patience} epochs")
            break
      
      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)
      results["F1"].append(f1)
      results["Precision"].append(precision_macro)
      results["Recall"].append(recall_macro)

      # log metrics to wandb
      wandb.log({"train_acc": train_acc, "train_loss": train_loss})
      wandb.log({"test_acc": test_acc, "test_loss": test_loss})
      wandb.log({"F1": f1, "Precision": precision_macro, "Recall": recall_macro})
      wandb.log({"confusion_matrix": wandb.Image(cfm_name)})
      wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})
      wandb.log({"Classification Report":df})

  # Return the filled results at the end of the epochs
  return results
