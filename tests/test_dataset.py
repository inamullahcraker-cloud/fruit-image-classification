from src.datasets.builder import build_dataloaders


def test_build_dataloaders(config):

    (
        train_loader,
        val_loader,
        test_loader,
        classes,
        class_to_idx,
    ) = build_dataloaders(config)

    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    assert len(test_loader.dataset) > 0

    assert len(classes) > 0

    assert isinstance(
        class_to_idx,
        dict,
    )
