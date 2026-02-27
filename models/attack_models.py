import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p) = -α(1-p)^γ log(p)
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        focal_loss = focal_weight * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EnhancedMIAFC(nn.Module):
    """Enhanced MIA attack model with deeper network and BatchNorm"""
    def __init__(self, input_dim=10, output_dim=1, hidden_dims=[512, 256, 128, 64], 
                 dropout=0.3, use_bn=True, use_residual=False):
        super(EnhancedMIAFC, self).__init__()
        self.use_residual = use_residual and (input_dim == hidden_dims[-1])
        
        layers = []
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        if self.use_residual:
            self.residual_proj = nn.Linear(input_dim, hidden_dims[-1])
    
    def forward(self, x):
        identity = x
        x = self.feature_layers(x)
        if self.use_residual:
            x = x + self.residual_proj(identity)
        x = self.output_layer(x)
        return x


class AttentionMIAFC(nn.Module):
    """MIA attack model with self-attention mechanism"""
    def __init__(self, input_dim=10, output_dim=1, hidden_dim=256, dropout=0.2):
        super(AttentionMIAFC, self).__init__()
        
        self.feature_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        features = self.feature_embed(x)
        attention_weights = torch.softmax(self.attention(features), dim=0)
        weighted_features = features * attention_weights
        output = self.classifier(weighted_features)
        return output


class MIAFC(nn.Module):
    """Original baseline MIA attack model (simple MLP without BatchNorm)"""
    def __init__(self, input_dim=10, output_dim=1, dropout=0.2):
        super(MIAFC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MIAFCBN(nn.Module):
    """Improved MIA attack model with BatchNorm"""
    def __init__(self, input_dim=10, output_dim=1, dropout=0.3):
        super(MIAFCBN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


class MIATransformer(nn.Module):
    """
    Transformer-based MIA attack model with flexible input dimensions.
    Treats each feature as a token in a sequence.
    """
    def __init__(self, input_dim=10, output_dim=1, hidden_dim=64, num_layers=3, nhead=4, dropout=0.2):
        super(MIATransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection: map each feature to hidden_dim
        self.input_projection = nn.Linear(1, hidden_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        
        # Positional encoding for feature positions
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, 
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        
        # Normalize input
        x = self.bn(x)  # (batch_size, input_dim)
        
        # Reshape to treat each feature as a token
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        
        # Project each feature to hidden_dim
        x = self.input_projection(x)  # (batch_size, input_dim, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding  # (batch_size, input_dim, hidden_dim)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, input_dim, hidden_dim)
        
        # Global average pooling over sequence dimension
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, input_dim)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, hidden_dim)
        
        # Output layers
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
