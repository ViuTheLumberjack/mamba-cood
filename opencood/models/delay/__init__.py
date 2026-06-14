
#from opencood.models.delay.cnn_delay import DelayModule
#from opencood.models.delay.delay_film import FeatureModifier
#from opencood.models.delay.delay_local_att import AttentionBasedModifier
#from opencood.models.delay.delay_uTransformer import UTransformerModifier
#from opencood.models.delay.delay_convlstm import FutureFramePredictor
from opencood.models.delay.delay_3dcnn import FutureFramePredictor
from opencood.models.delay.delay_mamba import MambaFutureFramePredictor
from opencood.models.delay.delay_multipred_mamba import MambaMultiPredictor
from opencood.models.delay.delay_multipred_mamba_no_lin import MambaMultiPredictorNoLin
from opencood.models.delay.delay_multipred_mamba_unet import MambaUNet
from opencood.models.delay.delay_multipred_mamba_lru import MambaLRU, MambaLRUPredictor
#from opencood.models.delay.delay_f2f import FutureFramePredictor
#from opencood.models.delay.delay_timesformer_style import FutureFramePredictor
#from opencood.models.delay.delay_transformer import FutureFramePredictor

__all__ = {
    #'CNNCompression': DelayModule,
    #'FiLM': FeatureModifier,
    '3DCNN': FutureFramePredictor,
    'Mamba': MambaFutureFramePredictor,
    'MambaMultiPredictor': MambaMultiPredictor,
    'MambaMultiPredictorNoLin': MambaMultiPredictorNoLin,
    'MambaUNet': MambaUNet,
    'MambaLRU': MambaLRU,
    # 'F2FDataset': F2FDataset
}

def build_delay_module(delay_cfg):
    delay_name = delay_cfg['core_method']
    error_message = f"{delay_name} is not found. " \
                    f"Please add your delay module's name in opencood/" \
                    f"models/delay/__init__.py"
    assert delay_name in ['3DCNN', 'Mamba', 'MambaMultiPredictor', 'MambaMultiPredictorNoLin', 'MambaUNet', 'MambaLRU'], error_message

    delay = __all__[delay_name](
        delay_cfg['args'],
    )

    return delay


