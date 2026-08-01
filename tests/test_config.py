from src.config.config import load_config


def test_load_config():
    """
    Test configuration loading.
    """

    config = load_config(
        "configs/config.yaml"
    )

    assert config is not None
    assert config.training.batch_size > 0
    assert config.training.epochs > 0
    assert config.model.name != ""
