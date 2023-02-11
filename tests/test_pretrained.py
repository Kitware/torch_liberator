from torch_liberator.initializer import Pretrained
import ubelt as ub
import torch


class CustomModel(torch.nn.Module):
    def __init__(self, classes=1000, width_per_group=64):
        super().__init__()
        import torchvision
        self.module = torchvision.models.resnet50(num_classes=classes, width_per_group=width_per_group)
        self.extra = torch.nn.Linear(1, 1)


class SuperCustomModel(torch.nn.Module):
    def __init__(self, classes=1000, width_per_group=64):
        super().__init__()
        self.orig = CustomModel(classes=classes, width_per_group=width_per_group)
        self.extra3 = torch.nn.Linear(3, 5)


def test_pretrained_with_torch_checkpoint():
    # TODO: add ability to ingest torch packages.
    try:
        import torchvision  # NOQA
    except ImportError:
        import pytest
        pytest.skip('no torchvision')

    dpath = ub.Path.appdir('torch_liberator/unittests/pretrained').ensuredir()
    model1 = CustomModel()
    model_state_dict = model1.state_dict()
    checkpoint_fpath = dpath / 'checkpoint.ckpt'
    with open(checkpoint_fpath, 'wb') as file:
        torch.save(model_state_dict, file)

    # Construct the initializer that will try and force these
    # checkpoint weights into some other model.
    initializer = Pretrained(checkpoint_fpath, association='embedding')

    # Basic test: load the weights into something exactly the same from a checkpoint path
    model2_v0 = CustomModel()
    info = initializer.forward(model2_v0)

    # Advanced setting 1: the number of classes changed
    model2_v1 = CustomModel(classes=20)
    info = initializer.forward(model2_v1)

    # Advanced setting 2: the model shrunk
    model2_v2 = CustomModel(classes=20, width_per_group=32)
    info = initializer.forward(model2_v2)

    # Advanced setting 3: the model grew
    model2_v3 = CustomModel(classes=1001, width_per_group=128)
    info = initializer.forward(model2_v3)

    # Advanced setting 4: The model is a subtree of the original model
    model3_v4 = CustomModel(classes=1001, width_per_group=128).module
    info = initializer.forward(model3_v4)

    # Advanced setting 5: The model is a supertree of the original model
    model3_v5 = SuperCustomModel(classes=1001, width_per_group=128)
    info = initializer.forward(model3_v5)

    del info
