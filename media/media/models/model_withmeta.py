import torch
import torch.nn as nn
import timm
    
def _model(model_dir, num_classes=13):
        print(model_dir)
        checkpoint = torch.load(model_dir)
        state_dict = checkpoint['state_dict']
        model = timm.create_model('convnext_tiny_in22k', pretrained=False, num_classes=6)
        model.load_state_dict({k[len('model.'):]: v for k, v in state_dict.items() if not k.startswith('model.classifier')})
        model.reset_classifier(0)
        return model

class ConvNextWithMetadata(nn.Module):
    def __init__(self, model_dir, num_metadata_features, num_classes):
        super(ConvNextWithMetadata, self).__init__()
        self.model = _model(model_dir, num_classes)
        self.metadata_fc = nn.Linear(num_metadata_features, 32)  # Process metadata
        self.classifier = nn.Sequential(
            nn.Linear(self.model.num_features + 32, 16),  # Concatenated features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):  
        # print('forward')
        # print(x)      
        x, metadata = x
        # print(x.shape)
        # print(metadata)
        x = self.model(x)  # Image features
        metadata = self.metadata_fc(metadata)  # Processed metadata
        x = torch.cat((x, metadata), dim=1)  # Concatenate
        x = self.classifier(x)
        return x

