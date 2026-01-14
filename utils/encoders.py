import functools
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP

    
import jax
import flaxmodels  # pip install flaxmodels
import numpy as np



class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out

class RobotResNet(nn.Module):
    """
    A wrapper around flaxmodels.ResNet18/50.
    1. Loads a pretrained backbone.
    2. Extracts the final feature map.
    3. Applies SpatialSoftmax.
    4. Projects to a user-defined embedding size.
    """
    output_dim: int          # Size of the final embedding (e.g., 256)
    backbone_name: str = 'resnet18' # 'resnet18' or 'resnet50'
    use_spatial_softmax: bool = True
    freeze_backbone: bool = True    # If True, stops gradients from flowing into ResNet

    @nn.compact
    def __call__(self, x, train: bool = True):
        # 1. Select and Initialize Backbone
        # 'pretrained="imagenet"' ensures weights are loaded during init
        if self.backbone_name == 'resnet18':
            backbone = flaxmodels.ResNet18(output='activations', pretrained='imagenet')
        elif self.backbone_name == 'resnet50':
            backbone = flaxmodels.ResNet50(output='activations', pretrained='imagenet')
        else:
            raise ValueError("Supported backbones: resnet18, resnet50")

        # 2. Forward Pass (Backbone)
        # Use train=False if freezing to keep Batch Norm statistics stable
        backbone_train_mode = train and not self.freeze_backbone
        
        # Get dictionary of feature maps
        features_dict = backbone(x, train=backbone_train_mode)
        
        # Extract the last spatial feature map.
        # For ResNet18/34 in flaxmodels, the last block is usually 'block4_1'.
        # For ResNet50, it is usually 'block4_2'. 
        # To be safe, we grab the last value from the dictionary.
        features = list(features_dict.values())[-2]

        print("Last feature shape", features.shape)
        
        # [Optional] Stop Gradient to freeze weights strictly
        if self.freeze_backbone:
            features = jax.lax.stop_gradient(features)

        features = nn.Dense(self.output_dim)(features)

        if len(features.shape) == 3:
            features = features.reshape(-1, self.output_dim)
        else:
            batch_size = features.shape[0]
            features = features.reshape(batch_size, -1, self.output_dim)        
        return features

# -----------------------------------------------------------------------------
# 3. Main Execution / Testing
# -----------------------------------------------------------------------------
encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    'resnet18_freeze': functools.partial(RobotResNet, backbone_name='resnet18', freeze_backbone=True),
    'resnet18': functools.partial(RobotResNet, backbone_name='resnet18', freeze_backbone=False),
}


def main():
    # Configuration
    OUTPUT_DIM = 512
    IMAGE_SIZE = 224
    BATCH_SIZE = 2
    
    print(f"Creating model with Output Dim: {OUTPUT_DIM}...")

    # Initialize Model
    model = RobotResNet(
        output_dim=OUTPUT_DIM, 
        backbone_name='resnet18', 
        freeze_backbone=True
    )

    # Create dummy input
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    # Initialize Variables (This will download ImageNet weights automatically)
    print("Initializing variables (downloading weights if needed)...")
    variables = model.init(key, dummy_input)

    # Run Inference
    print("Running forward pass...")
    output = model.apply(variables, dummy_input, train=False)

    # Check Dimensions
    print("-" * 30)
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print("-" * 30)

if __name__ == "__main__":
    main()