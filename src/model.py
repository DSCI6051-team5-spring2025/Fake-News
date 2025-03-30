import torch.nn as nn
from transformers import CLIPModel

class CLIPClassifier(nn.Module):
    def __init__(self):
        super(CLIPClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.classifier = nn.Linear(self.clip.config.projection_dim, 2)

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
        outputs = self.clip(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, return_loss=False)
        pooled_output = outputs[0]  # logits
        return self.classifier(pooled_output), pooled_output
