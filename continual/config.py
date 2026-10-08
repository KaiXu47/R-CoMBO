from detectron2.config import CfgNode as CN


def add_continual_config(cfg):
    cfg.CONT = CN()
    cfg.CONT.TOT_CLS = 150
    cfg.CONT.BASE_CLS = 100
    cfg.CONT.INC_CLS = 10
    cfg.CONT.SETTING = 'overlapped'
    cfg.CONT.TASK = 1
    cfg.CONT.WEIGHTS = None
    cfg.CONT.OLD_WEIGHTS = None
    cfg.CONT.MED_TOKENS_WEIGHT = 3.0
    cfg.CONT.USE_OLD_MODEL = True
    cfg.CONT.LOGIT_DISTILL_WEIGHT = 5.0
    cfg.CONT.FEATURE_DISTILL_WEIGHT = 5.0
    cfg.CONT.MEMORY = False
    cfg.CONT.RANK = 16
    cfg.CONT.TRAIN_ACCUMULATED_CLASSES = False
    cfg.CONT.SEMANTIC_OLD_TO_IGNORE = False
    cfg.CONT.VOC_BACKGROUND_FOLDING = False
    cfg.CONT.VOC_FUTURE_TO_BACKGROUND = False
    cfg.CONT.VOC_FUTURE_PROXY = False
    cfg.CONT.VOC_FUTURE_PROXY_INIT = False
    # Rank previous-stage structure queries to route semantic gradients.
    cfg.CONT.IMPORTANCE_GRAD_ROUTING = True
    cfg.CONT.IMPORTANCE_GRAD_MIN = 0.1
    # Read detached image features within the predicted structure masks.
    cfg.CONT.SEMANTIC_PIXEL_CROSS_ATTENTION = True
    cfg.CONT.DETACH_PIXEL_FEATURES = True
    # Optional replay support for VOC.
    cfg.CONT.VOC_BOUNDARY_REPLAY = False
    cfg.CONT.VOC_BOUNDARY_REPLAY_MAX_IMAGES_PER_CLASS = 50
    cfg.CONT.VOC_REPLAY_OLD_CLASS_WEIGHT = 1.0
    cfg.CONT.VOC_REPLAY_SAMPLE_RATIO = 3
