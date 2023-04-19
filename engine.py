"""
Contains functions for training and testing a PyTorch model.

"""
import torch
import wandb
from tqdm.auto import tqdm



def train_step(model, dataloader, loss_fn, optimizer, device):

  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # 6. Scheduler step
     # scheduler.step()
      
      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  
  return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):
  
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

def wandbinit(lr=0.0001, architecture = "CNN", dataset = "Custom FSL Dataset", epoch = 20, optimizer = "RMSProp"):
           # start a new wandb run to track this script
  wandb.init(
  # set the wandb project where this run will be logged
  project="FILOSIGN-quantized",
        
  # track hyperparameters and run metadata
  config={
  "learning_rate": lr,
  "architecture": architecture,
  "dataset": dataset,
  "epochs": epoch,
  "optimizer": optimizer
        }
    )


def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device, lr, architecture, dataset, opt_name, model_path):
  
    
  wandbinit(lr, architecture, dataset, epochs, opt_name)
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }
    
  best_test_acc = 0
  test_acc = 0
  model_save_path = model_path

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )
        

    # Check if test accuracy is better than current best
      if test_acc > best_test_acc:
        # If so, save model to path
            torch.save(obj=model.state_dict(), f= model_save_path)

            # Update best test accuracy
            best_test_acc = test_acc

            # Save the model state_dict()
            print(f"[INFO] Saving model to: {model_save_path} with test accuracy: {test_acc:.4f}")

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

      # log metrics to wandb
      wandb.log({"train_acc": train_acc, "train_loss": train_loss})
      wandb.log({"test_acc": test_acc, "test_loss": test_loss})

  # Return the filled results at the end of the epochs
  return results