__version__ = '0.1.dev0'

import os
if "S2S_DIR" not in os.environ:
    s2s_dir = os.path.abspath(os.path.join(__file__,"../../.."))
    os.environ["S2S_DIR"] = s2s_dir

