## Using the custom architecture with Ultralytics

This model (`weights/best.pt`) requires a few custom modules (QuantumConv, NeuroMorph, WaveletProc, MultiModalFusion).
To use the model:

1. Install Ultralytics:

2. Add Custom Modules To The Existing Blocks:

\ultralytics\ultralytics\nn\modules\block.py

Add these

'''
__all__ = (
    "QuantumConv", 
    "NeuroMorph", 
    "WaveletProc", 
    "MultiModalFusion",
    "Upsample", 
    "C3k2", 
    "SPPF", 
    "C2PSA", 
    "QuantumDetect"
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
)

class QuantumConv(nn.Module):
    """Quantum-inspired Convolutional Block"""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class NeuroMorph(nn.Module):
    """Neuromorphic Processing Block"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class WaveletProc(nn.Module):
    """Wavelet Processing Block"""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MultiModalFusion(nn.Module):
    """Multi-Modal Fusion Block"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        # c1 should be sum of input channels when concatenating
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        
    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.act(self.bn(self.conv(x)))

class Upsample(nn.Module):
    """Upsampling Block"""
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class C3k2(nn.Module):
    """Simplified C3 Block for YAML compatibility"""
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut) for _ in range(n)))
        
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class C2PSA(nn.Module):
    """C2 Block with PSA (Pixel-Shuffle Attention) inspired design"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
        
    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))

class QuantumDetect(nn.Module):
    """Quantum-inspired Detection Head"""
    def __init__(self, c1, c2, num_classes=80, anchors=1):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        
        # Detection convolution
        self.conv = nn.Conv2d(c1, anchors * (5 + num_classes), 1)
        
    def forward(self, x):
        return self.conv(x)
        
'''

3. Initialize The Custom Blocks:

\ultralytics\ultralytics\nn\modules\__init__.py

'''

from .block import (
    QuantumConv,
    NeuroMorph,
    WaveletProc,
    MultiModalFusion,
    Upsample,
    C3k2,
    SPPF,
    C2PSA,
    QuantumDetect,
    C1,
    C2,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    MaxSigmoidAttnBlock,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
)

'''

4. Update The Architecture Head: 

ultralytics\ultralytics\cfg\models\11\yolo11.yaml

'''

# yolo11_quantum_working.yaml
nc: 4
scales:
  n: [0.33, 0.25, 1024]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, QuantumConv, [64, 3, 2]]    # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]          # 1-P2/4
  - [-1, 2, C2f, [128]]                 # 2
  - [-1, 1, WaveletProc, [256, 3, 2]]   # 3-P3/8
  - [-1, 2, C2f, [256]]                 # 4
  - [-1, 1, Conv, [512, 3, 2]]          # 5-P4/16
  - [-1, 2, C2f, [512]]                 # 6
  - [-1, 1, NeuroMorph, [1024, 3, 2]]   # 7-P5/32
  - [-1, 2, C2f, [1024]]                # 8
  - [-1, 1, SPPF, [1024, 5]]            # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 10
  - [[-1, 6], 1, Concat, [1]]                   # 11 cat backbone P4
  - [-1, 2, C2f, [512]]                         # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 13
  - [[-1, 4], 1, Concat, [1]]                   # 14 cat backbone P3
  - [-1, 2, C2f, [256]]                         # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]                  # 16
  - [[-1, 12], 1, Concat, [1]]                  # 17 cat head P4
  - [-1, 2, C2f, [512]]                         # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]                  # 19
  - [[-1, 9], 1, Concat, [1]]                   # 20 cat head P5
  - [-1, 2, C2f, [1024]]                        # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]             # 22 Detect(P3, P4, P5)

  '''
