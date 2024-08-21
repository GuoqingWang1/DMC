import pandas as pd

class Discretizer(object):
    def __init__(self):
        self.discre_types = ["equal_width"]

    def apply(self, num_bins, data, discre_type):
        assert discre_type in self.discre_types, "discre_type is not available!"

        if discre_type == "equal_width":
            data = pd.Series(data)
            bins = pd.cut(data, bins=num_bins, labels=[k+1 for k in range(num_bins)])
            return list(bins)
        