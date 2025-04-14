from transformers import BlipModel
import torch.nn as nn

class BlipForFakeNewsClassification(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.blip = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        self.classifier = nn.Linear(self.blip.config.projection_dim, num_labels)

    def forward(self, input_ids, pixel_values, attention_mask=None):
        outputs = self.blip(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        pooled_output = outputs.image_embeds
        return self.classifier(pooled_output)