import torch.nn as nn
from transformers import CLIPModel

class CLIPClassifier(nn.Module):
    def __init__(self):
        super(CLIPClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.classifier = nn.Linear(self.clip.config.projection_dim, 2)

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
        text_features = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask) if input_ids is not None else None
        image_features = self.clip.get_image_features(pixel_values=pixel_values) if pixel_values is not None else None

        if text_features is not None and image_features is not None:
            fused = (text_features + image_features) / 2
        elif text_features is not None:
            fused = text_features
        else:
            fused = image_features

        return self.classifier(fused), fused
