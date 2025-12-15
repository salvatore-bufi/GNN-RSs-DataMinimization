def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break

import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .ngcf.NGCF import NGCF
        from .knngnn import KNNGNN
        from .bprmf import BPRMF
        # from .itemknngnn import ITEMKNNGNN
        # from .hifGem import HIFGem
        # from .lightgcnbias import LightGCNbias
        # from .graphILFcos import graphILFcos
        # from .ILFgraph import ILFgraph
        # from .GNeNe import GNeNe
        # from .mfgssm import MFGSSM
        # from .reverseGNN import reverseGNN
        # from .bprmfMeanUI import BPRMFmeanUI
        from .lightgcn.LightGCN import LightGCN
        from .dgcf.DGCF import DGCF
        from .sgl.SGL import SGL
        from .ultragcn import UltraGCN
        from .sgl import SGL
        from .simgcl import SimGCL
        #
        # from .hif import HIF
        from .SimpleX import SimpleX
        from .DirectAU import DirectAU
        from .svd_gcn import SVDGCN
        from .gfcf import GFCF
