from PIL import Image
import operator
from .mytypes import ItemFilter


def true_filter(item) -> bool:
    return True


class SizeFilter:  # type: ItemFilter

    def __init__(self, min_size: int = 25):
        self.min_size = min_size

    def __call__(self, item):
        size = Image.open(str(item[0])).size
        return min(*size) > self.min_size


class ExistsFilter:  # type: ItemFilter

    def __call__(self, item):
        return item[0].exists()


class ComposeFilter:  # type: ItemFilter

    def __init__(self, filters, logical_op=operator.and_):
        self.reversed_filters = list(reversed(filters))
        self.logical_op = logical_op

    def __call__(self, item):
        res = self.reversed_filters[0](item)
        for cur_filter in self.reversed_filters[1:]:
            res = self.logical_op(cur_filter(item), res)
        return res
