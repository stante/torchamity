def accuracy(pred, y, result):
  def _accuracy(pred, y, result=None):
    return (pred == y).float().mean().item()
  
  acc = _accuracy(pred, y)

  if 'acc' not in result:
    result['acc'] = []

  result['acc'].append(acc)
