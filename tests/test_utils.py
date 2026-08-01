from src.utils.device import get_device


def test_device():

    device = get_device("cpu")

    assert str(device) == "cpu"
