import numpy as np
from live_filter import Filters


def test_kernel_implementation():
    f = Filters()
    assert 8 <= len(f.kernels)


def test_apply_filter():
    f = Filters()
    f.kernels["test_filter"] = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
    fake_frame = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    filtered = f.apply_filter(fake_frame, "test_filter")
    assert np.array_equal(filtered, fake_frame * 2)


def test_switch_next():
    f = Filters()
    new_kernels = {}
    new_kernels["first"] = 1
    new_kernels["second"] = 2
    new_kernels["third"] = 3
    new_kernels["fourth"] = 4
    new_kernels["fifth"] = 5
    f.kernels = new_kernels

    # Get initial kernel
    assert "first" == f.get_current_filter_name()

    # Switch to the next
    f.switch_next_filter()
    assert "second" == f.get_current_filter_name()

    # Circular test
    f.switch_next_filter()  # Third
    f.switch_next_filter()  # Fourth
    f.switch_next_filter()  # Fifth
    f.switch_next_filter()  # First
    assert "first" == f.get_current_filter_name()

    # Switch to the previous (circular)
    f.switch_previous_filter()  # Fifth
    assert "fifth" == f.get_current_filter_name()

    f.switch_next_filter()  # Fourth
    f.switch_next_filter()  # Third
    f.switch_next_filter()  # Second
    f.switch_next_filter()  # First
    f.switch_next_filter()  # Fifth
    assert "fifth" == f.get_current_filter_name()
