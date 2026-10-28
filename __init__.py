"""StyleGAN2-Sobieszek: accelerated StyleGAN2 synthesis on CUDA and MPS.

Derived from NVIDIA StyleGAN3 / StyleGAN2-ADA (``torch_utils``, ``dnnlib``)
with shared-convolution/polyphase synthesis, Metal kernels and the
``pointwise_style32`` early-output generators. See README.md for usage and
LICENSE.txt for terms.

NVIDIA pickles and this engine's own modules import ``torch_utils`` and
``dnnlib`` by absolute name, so this directory is appended to ``sys.path``.
"""

import sys
from pathlib import Path

_ENGINE_DIR = str(Path(__file__).resolve().parent)
if _ENGINE_DIR not in sys.path:
    sys.path.append(_ENGINE_DIR)
