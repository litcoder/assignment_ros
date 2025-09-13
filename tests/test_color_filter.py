from src.color_filter import Filters


def test_kernels():
    f = Filters()

    # convert to lower case for string comp.
    available_filters = f.get_filter_names()
    available_filters = list(map(lambda x: x.lower(), available_filters))

    assert 8 <= len(available_filters), "모든 필터들이 정의되어야 합니다"

    # Case insenstive string comparison.
    assert "Original".lower() in available_filters, "Orinal filter"
    assert "Blur".lower() in available_filters, "Blur filter"
    assert "Gausian blur".lower() in available_filters, "Gausian blur filter"
    assert "Sharpen".lower() in available_filters, "Sharpen filter"
    assert "Sobel (x)".lower() in available_filters, "Sobel (X) filter"
    assert "Sobel (y)".lower() in available_filters, "Sobel (y) filter"
    assert "Edge detection".lower() in available_filters, "Edge detection filter"
