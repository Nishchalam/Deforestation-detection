import pytest
import torch
from src.data.dataset import EuroSATDataset

@pytest.fixture
def mock_csv(tmp_path):
    csv_file = tmp_path / "train.csv"
    csv_file.write_text("image_path,label,class_name\ndummy.jpg,1,Forest\ndummy2.jpg,0,NonForest")
    return csv_file

@pytest.fixture
def mock_image(tmp_path):
    from PIL import Image
    import numpy as np
    
    img_path = tmp_path / "dummy.jpg"
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    img.save(img_path)
    
    img_path2 = tmp_path / "dummy2.jpg"
    img2 = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    img2.save(img_path2)
    return tmp_path

def test_dataset_length(mock_csv, mock_image):
    dataset = EuroSATDataset(csv_file=mock_csv, root_dir=mock_image)
    assert len(dataset) == 2

def test_dataset_item(mock_csv, mock_image):
    dataset = EuroSATDataset(csv_file=mock_csv, root_dir=mock_image)
    item = dataset[0]
    
    assert "image" in item
    assert "label" in item
    assert "class_name" in item
    # Since we didn't pass transform, it remains a PIL Image
    # Or if there's a default, check if it's tensor
    from PIL import Image
    assert isinstance(item["image"], Image.Image) or isinstance(item["image"], torch.Tensor)
