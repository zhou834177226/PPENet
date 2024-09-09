from .GEMNet import build_GEMNet

_ZSL_META_ARCHITECTURES = {
    "GEMModel": build_GEMNet,
}

def build_zsl_pipeline(cfg):
    meta_arch = _ZSL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    # 记录了函数名，返回时直接调用build_GEMNet
    return meta_arch(cfg)