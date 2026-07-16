import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from src.explainability import (
    GradCAM,
    GradCAMPlusPlus,
    GuidedBackprop,
    SaliencyMap,
    FeatureMapExtractor,
    OcclusionSensitivity,
    LIMESimulator,
    preprocess_image,
    find_last_conv_layer,
    generate_explainability_dashboard
)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def test_explainability_suite(tmp_path):
    model = DummyModel()
    device = torch.device("cpu")
    
    # 1. Preprocess
    img = Image.new("RGB", (64, 64), color="green")
    tensor = preprocess_image(img, device)
    
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.requires_grad
    
    # Target layer
    target_layer = find_last_conv_layer(model)
    assert target_layer is model.conv2
    
    # 2. Grad-CAM
    cam_generator = GradCAM(model, target_layer)
    cam = cam_generator.generate_heatmap(tensor, class_idx=0)
    assert cam.shape == (224, 224)
    assert 0.0 <= np.min(cam) <= np.max(cam) <= 1.0
    cam_generator.remove_hooks()
    
    # 3. Grad-CAM++
    cam_pp_generator = GradCAMPlusPlus(model, target_layer)
    cam_pp = cam_pp_generator.generate_heatmap(tensor, class_idx=0)
    assert cam_pp.shape == (224, 224)
    cam_pp_generator.remove_hooks()
    
    # 4. Saliency
    saliency = SaliencyMap(model)
    sal = saliency.generate_saliency(tensor, class_idx=0)
    assert sal.shape == (224, 224)
    
    # 5. Guided Backpropagation
    # Refresh gradient tracking tensor
    tensor = preprocess_image(img, device)
    gb = GuidedBackprop(model)
    grads = gb.generate_gradients(tensor, class_idx=0)
    assert grads.shape == (224, 224, 3)
    gb.remove_hooks()
    
    # 6. Feature Maps
    extractor = FeatureMapExtractor(model)
    feats = extractor.extract_features(tensor, target_layer)
    assert len(feats.shape) == 3
    assert feats.shape[0] == 16 # channels
    
    # 7. Occlusion Sensitivity
    occ = OcclusionSensitivity(model)
    occ_map = occ.generate_sensitivity_map(tensor, class_idx=0, box_size=32, stride=16)
    assert occ_map.shape == (224, 224)
    
    # 8. LIME simulation
    lime = LIMESimulator(model)
    lime_map = lime.generate_lime_attribution(tensor, class_idx=0, grid_dim=4, num_perturbations=10)
    assert lime_map.shape == (224, 224)
    
    # 9. Dashboard Plot
    save_path = tmp_path / "dashboard.png"
    generate_explainability_dashboard(
        img,
        sal,
        occ_map,
        cam,
        cam_pp,
        lime_map,
        save_path
    )
    assert save_path.exists()
