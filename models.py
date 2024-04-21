import torch

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
    
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        self.main= torch.nn.Sequential(
            
            torch.nn.Conv2d(in_channels= input_channels, out_channels= 128, kernel_size= 4, stride= 2,  padding= 1),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Conv2d(in_channels= 128, out_channels= 256, kernel_size= 4, stride= 2,  padding= 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Conv2d(in_channels= 256, out_channels= 512, kernel_size= 4, stride= 2,  padding= 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Conv2d(in_channels= 512, out_channels= 1024, kernel_size= 4, stride= 2,  padding= 1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Conv2d(in_channels= 1024, out_channels= 1, kernel_size= 4, stride= 2,  padding= 0)
            
        ) 
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        
        ##########       END      ##########
        
        return self.main(x)


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        self.main= torch.nn.Sequential(
            
            torch.nn.ConvTranspose2d(in_channels= noise_dim, out_channels= 1024, kernel_size= 4, stride= 1, padding= 0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.ConvTranspose2d(in_channels= 1024, out_channels= 512, kernel_size= 4, stride= 2, padding= 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.ConvTranspose2d(in_channels= 512, out_channels= 256, kernel_size= 4, stride= 2, padding= 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.ConvTranspose2d(in_channels= 256, out_channels= 128, kernel_size= 4, stride= 2, padding= 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.ConvTranspose2d(in_channels= 128, out_channels= 3, kernel_size= 4, stride= 2, padding= 1),
            torch.nn.Tanh()
            
        )
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        
        ##########       END      ##########
        
        return self.main(x)
