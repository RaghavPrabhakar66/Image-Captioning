from PIL import Image
import torch as T
from torchvision import transforms
from models import Model
from utils import show_imgs, Vocab

class Inference:
    def __init__(self, size, model):
        self.inference_transforms = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize(size)])
        self.model = model
        self.token2word = Vocab().index2word

    # Predicts caption for a list of images    
    def run(self, images, show=False):
        # Convert images to tensors for evaluation
        images = T.stack([self.inference_transforms(image) for image in images])

        # Model outputs tokens
        predicted_tokens = self.model.predict(images, 37).cpu().numpy()
        # Convert tokens to words and display corresponding images        
        predicted_captions = [' '.join([self.token2word[i] for i in caption]) for caption in predicted_tokens]
        if show:
            for image in images:
                show_imgs(image)
        
        return predicted_captions

if __name__ == '__main__':
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
    image = Image.open('Enter_Image_Name')
    print(inf.run([image], show=False))
