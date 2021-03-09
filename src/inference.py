from PIL import Image
import torch as T
from torchvision import transforms

class Inference:
    def __init__(self, size, model, index_to_word):
        self.inference_transforms = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize(size)])
        self.model = model
        self.token2word = index_to_word

    # Predicts caption for a list of images    
    def run(self, images):
        # Convert images to tensors for evaluation
        images = T.stack([self.inference_transforms(image) for image in images])

        # Model outputs tokens
        predicted_tokens = self.model.predict(images, 37).cpu().numpy()
        # Convert tokens to words      
        predicted_captions = [' '.join([self.token2word[i] for i in caption]) for caption in predicted_tokens]
       
        return predicted_captions
