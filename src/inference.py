from PIL import Image
import torch as T
from torchvision import transforms
from models import Model
from utils import show_imgs

class Inference:
    def __init__(self, size, model):
        self.inference_transforms = transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Resize(size)
        ])
        self.model = model
    
    def run(self, images, show=False):
        images = T.stack([self.inference_transforms(image) for image in images])
        predicted_tokens = self.model.predict(images, 37)
        if show:
            show_imgs(image)
        return predicted_tokens


model = Model(
            backbone='vgg16',
            embed_size=128, 
            hidden_size=128, 
            vocab_size=37, 
            lstm_cells=128, 
            lstm_dropout=0.5,
            verbose=False,
            device=T.device('cuda:0'))

inf = Inference((244, 244), model)
image = Image.open('src/image.jfif')
print(inf.run([image], show=False))
