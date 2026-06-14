from .convolutional import Encoder, Decoder
from .conv_glu import ConvGLUEncoder
from .mamba_glu import MambaGLUEncoder, MambaGLUDecoder
from .vit_encoder import VideoViTEncoderWrapper
from .dino_encoder import PretrainedDinoEncoder
from .linear import LinearEncoder, LinearDecoder

__all_encoders__ = ['ConvGLUEncoder', 'MambaGLUEncoder', 'VideoViTEncoderWrapper', 'PretrainedDinoEncoder']
__all_decoders__ = ['Decoder', 'MambaGLUDecoder', 'LinearDecoder']

def get_encoder(encoder_args):
    match encoder_args.get('type', 'mambaglu'):
        case 'vit':
            return VideoViTEncoderWrapper(encoder_args)
        case 'dino':
            return PretrainedDinoEncoder(encoder_args)
        case 'mambaglu':
            return MambaGLUEncoder(encoder_args)
        case "convglu":
            return ConvGLUEncoder(encoder_args)
        case "conv":
            return Encoder(encoder_args)
        case _:
            raise ValueError(f"Unknown encoder type : ", encoder_args)

def get_decoder(decoder_args):
    match decoder_args.get('type', 'conv'):
        case 'conv':
            return Decoder(decoder_args)
        case 'mambaglu':
            return MambaGLUDecoder(decoder_args)
        case "linear":
            return LinearDecoder(decoder_args)
        case _:
            raise ValueError(f"Unknown decoder type: ", decoder_args)