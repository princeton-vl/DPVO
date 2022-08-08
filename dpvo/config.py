from yacs.config import CfgNode as CN

_C = CN()

# max number of keyframes
_C.BUFFER_SIZE = 2048

# bias patch selection towards high gradient regions?
_C.GRADIENT_BIAS = True

# VO config (increase for better accuracy)
_C.PATCHES_PER_FRAME = 80
_C.REMOVAL_WINDOW = 20
_C.OPTIMIZATION_WINDOW = 12
_C.PATCH_LIFETIME = 12

# threshold for keyframe removal
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 12.5

# camera motion model
_C.MOTION_MODEL = 'DAMPED_LINEAR'
_C.MOTION_DAMPING = 0.5

_C.MIXED_PRECISION = True

cfg = _C
