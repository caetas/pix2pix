import torch

def create_encoder():
    # use a resnet-18 backbone
    encoder = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    # remove the last two layers
    encoder = torch.nn.Sequential(*(list(encoder.children())[:-2]))
    # freeze the encoder
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder

def create_decoder():
    # the decoder should take as an input of 1,512,8,8 and outputs an image of 256x256x3
    decoder = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        torch.nn.Sigmoid()
    )
    return decoder

def create_autoencoder():
    encoder = create_encoder()
    decoder = create_decoder()
    autoencoder = torch.nn.Sequential(
        encoder,
        decoder
    )
    return autoencoder