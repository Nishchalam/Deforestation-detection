import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from src.inference import PatchGenerator, LandCoverPredictor, LandCoverMapper

class DummyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def test_patch_generator():
    img = Image.new("RGB", (256, 256), color="green")
    generator = PatchGenerator(patch_size=64, stride=64)
    
    patches = []
    bboxes = []
    for patch_data in generator.extract_patches(img):
        patches.append(np.array(patch_data["image"]))
        bboxes.append(patch_data["bbox"])
        
    assert len(patches) == 16
    assert len(bboxes) == 16
    
    # Reconstruct
    recon = generator.reconstruct_image(patches, bboxes, (256, 256))
    assert recon.shape == (256, 256, 3)
    assert np.all(recon == np.array(img))

def test_land_cover_mapper(tmp_path):
    img = Image.new("RGB", (128, 128), color="blue")
    img_path = tmp_path / "test_sat.png"
    img.save(img_path)
    
    model = DummyCNN()
    # Mock state dict
    ckpt_path = tmp_path / "dummy_best.pth"
    torch.save(model.state_dict(), ckpt_path)
    
    predictor = LandCoverPredictor(model, str(ckpt_path))
    mapper = LandCoverMapper(predictor, patch_size=64, stride=64)
    
    outputs = mapper.generate_map(str(img_path), batch_size=2)
    
    assert "classes" in outputs
    assert len(outputs["classes"]) == 4
    assert outputs["prediction_map"].size == (128, 128)
    assert outputs["confidence_map"].size == (128, 128)
