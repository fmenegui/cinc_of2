import torch
import torch.nn as nn
import timm


class Metadata(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(Metadata, self).__init__()
        self.metadata_fc = nn.Linear(num_metadata_features, 32)  
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):        
        metadata = x
        metadata = self.metadata_fc(metadata)  # Processed metadata
        x = self.classifier(metadata)
        return x
    

class ConvNextWithMetadata(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(ConvNextWithMetadata, self).__init__()
        self.convnext = timm.create_model('convnext_tiny_in22k', pretrained=True, num_classes=0)  # Use 'num_classes=0' to remove the original classifier
        self.metadata_fc = nn.Linear(num_metadata_features, 32)  # Process metadata
        self.classifier = nn.Sequential(
            nn.Linear(self.convnext.num_features + 32, 16),  # Concatenated features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):        
        x, metadata = x
        x = self.convnext(x)  # Image features
        metadata = self.metadata_fc(metadata)  # Processed metadata
        x = torch.cat((x, metadata), dim=1)  # Concatenate
        x = self.classifier(x)
        return x
    
    
class ConvNextWithMetadataScratch(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(ConvNextWithMetadata, self).__init__()
        self.convnext = timm.create_model('convnext_tiny_in22k', pretrained=False, num_classes=0)  # Use 'num_classes=0' to remove the original classifier
        self.metadata_fc = nn.Linear(num_metadata_features, 32)  # Process metadata
        self.classifier = nn.Sequential(
            nn.Linear(self.convnext.num_features + 32, 16),  # Concatenated features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):        
        x, metadata = x
        x = self.convnext(x)  # Image features
        metadata = self.metadata_fc(metadata)  # Processed metadata
        x = torch.cat((x, metadata), dim=1)  # Concatenate
        x = self.classifier(x)
        return x

# Example
if __name__ == "__main__":
    model = ConvNextWithMetadata(num_metadata_features=4, num_classes=10)
    image = torch.randn(1, 3, 224, 224)  # Example image tensor
    metadata = torch.randn(1, 4)  # Example metadata tensor (age, gender, height, weight)
    output = model(image, metadata)
