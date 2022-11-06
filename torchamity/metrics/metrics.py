class Mean:
    """Generiic class that calculates the mean.
    """
    def __init__(self, name):
        self.name = name
        self.reset()

    def update(self, *args):
        raise NotImplementedError("Must be implemented in subclass.")

    def result(self, reset=True):
        if self.count == 0:
            return 0
        else:
            total = self.total
            count = self.count
            if reset:
                self.reset()
                
            return total / count

    def reset(self):
        self.total = 0
        self.count = 0


class Loss(Mean):
    """Calcualtes the loss metric.
    """
    def __init__(self, name='loss'):
        super().__init__(name)

    def update(self, loss):
        self.total += loss
        self.count += 1


class Accuracy(Mean):
    """Calculates accuracy metric.
    """
    def __init__(self, name='acc'):
        super().__init__(name)

    def update(self, y_true, y_pred):
        self.total += (y_true == y_pred).float().mean().item()
        self.count += 1
