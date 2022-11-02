class Accuracy:
  """Calculates accuracy metric.
  """
  def __init__(self, name='acc'):
    self.name = name
    self.reset()

  def update(self, y_true, y_pred):
    self.total += (y_true == y_pred).float().mean().item()
    self.count += 1

  def result(self):
    return self.total / self.count

  def reset(self):
    self.total = 0
    self.count = 0

