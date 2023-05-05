class VOC2007Metric(object):
    def __init__(self):
        self.top_1s, self.top_5s = list(), list()
        self.num_samples = list()

    def computeAccuracy(self, x, y, mode):
        _, x = x.topk(k=5, dim=1, largest=True, sorted=True)
        x = x.t()
        correct = x.eq(y.expand_as(other=x))

        acc = list()
        for k in (1, 5):
            correct_k = correct[:k].float().sum()
            acc.append(correct_k.mul_(100 / len(x.t())))
        
        if mode == 'test':
            self.top_1s.append(acc[0])
            self.top_5s.append(acc[1])
            self.num_samples.append(len(x.t()))

        return acc
    
    def computeAverage(self):
        acc = list(0 for i in range(2))
        for top_1, top_5, num_sample in zip(self.top_1s, self.top_5s, self.num_samples):
            acc[0] += top_1 * num_sample
            acc[1] += top_5 * num_sample
        for i in range(2):
            acc[i] /= sum(self.num_samples)
        
        return acc
    
    def printAccuracy(self, mode, epoch=None, batch=None, loss=None, acc=None):
        if mode == 'train':
            print('[Epoch] {}/{} [Batch] {}/{} [Loss] {:.4f} [Top-1] {:.2f} [Top-5] {:.2f}' \
                  .format(epoch[0], epoch[1], batch[0], batch[1], loss, acc[0], acc[1]))
        elif mode == 'test':
            print('[Batch] {}/{} [Top-1] {:.2f} [Top-5] {:.2f}'.format(batch[0], batch[1], acc[0], acc[1]))
        elif mode == 'end':
            acc = self.computeAverage()
            print('[Top-1] {:.2f} [Top-5] {:.2f}'.format(acc[0], acc[1]))
