def test_load_partial_state_with_module():
    import torch_liberator
    import torchvision
    import ubelt as ub

    # Create two similar but different models
    faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    model = torchvision.models.resnet50(pretrained=True)

    faster_rcnn.load_state_dict(model.state_dict())

    # Return a dictionary that tells what load_partial_state did
    info = torch_liberator.load_partial_state(
        faster_rcnn, model.state_dict(),
        association='isomorphism')

    print(ub.map_vals(len, info['seen']))
    print(ub.map_vals(len, ub.dict_diff(info, ['seen'])))
    assert info['seen']['full_add'] == 265
    assert info['seen']['skipped'] == 55
    assert info['other_unused'] == 55
    assert info['self_unset'] == 30
