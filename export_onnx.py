from config import *

if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint_.pth.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.cpu()
    # Input to the model
    x = torch.randn(batch_size, 3, 112, 112, requires_grad=True)
    filename = 'age_and_gender.onnx'
    torch.onnx._export(model, x, filename)
    print('Saved at {}'.format(filename))
