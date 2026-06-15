from models.segment_anything.build_sam import sam_model_registry
from models.segment_anything_samus.build_sam_us import samus_model_registry
from models.segment_anything_samus_autoprompt.build_samus import autosamus_model_registry
import os
from pathlib import Path

_CKPT_ROOT = Path(
    os.environ.get(
        "FETALAGENT_CKPT_DIR",
        str(Path(__file__).resolve().parents[4] / "FetalAgent_ckpt"),
    )
).resolve()

def get_model(modelname="SAM", args=None, opt=None):
    print('*'*100)
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint=str(_CKPT_ROOT / 'SAMUS.pth'))
    elif modelname == "SAMUS":
        model = samus_model_registry['vit_b'](checkpoint=str(_CKPT_ROOT / 'SAMUS.pth'))
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
