import numpy as np 

class AverageMeter(object):
    """Average Meter"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, key=None):
        if key is None:
            self.reset_all()
            return
        item = self.book.get(key, None)
        if item is not None:
            item[0] = 0 # value 
            item[1] = 0 # count
    
    def update(self, key, val):
        record = self.book.get(key, None)
        if record is None:
            self.book[key] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, key):
        record = self.book.get(key, None)
        assert record is not None
        return record[0] / record[1]


class StreamClsMetrics(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, pred, target):
        pred = pred.max(dim=1)[1].cpu().numpy().astype(np.uint8)
        target = target.cpu().numpy().astype(np.uint8)
        for lt, lp in zip(target, pred):
            self.confusion_matrix[lt][lp] += 1

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)
        return string

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) +
                              hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
