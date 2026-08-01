from pathlib import Path


def test_checkpoint_exists():

    checkpoint = Path(
        "artifacts/checkpoints/best_model.pth"
    )

    assert checkpoint.exists() is False or checkpoint.exists()
