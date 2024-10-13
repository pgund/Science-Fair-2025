import torch

def save_model(model, path='model.pth'):
    """ Save the model's state dict to a file """
    torch.save(model.state_dict(), path)

def load_model(model, path='model.pth'):
    """ Load the model's state dict from a file """
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()  # Switch to evaluation mode
    return model
