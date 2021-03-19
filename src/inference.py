import torch as T
from torchvision import transforms

class Inference:
    def __init__(self, size, model, index_to_word, maxlen):
        self.inference_transforms = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize(size)])
        self.model = model
        self.token2word = index_to_word
        self.maxlen = maxlen

    # Predicts caption for a list of images    
    def run(self, images, beam=None):
        # Convert images to tensors for evaluation
        images = T.stack([self.inference_transforms(image) for image in images])

        # Model outputs tokens
        predicted_tokens = self.model.predict(images, self.maxlen, beam).cpu().numpy()
        # Convert tokens to words      
        predicted_captions = [' '.join([self.token2word[i] for i in caption]) for caption in predicted_tokens]
       
        return predicted_captions
