from tqdm import tqdm

class Learner:
  def __init__(self, train, val, optim, loss_fn, model, val_metrics = None):
    self.train = train
    self.val = val
    self.optim = optim
    self.loss_fn = loss_fn
    self.model = model

  def fit(self, epochs):
    result = {'loss':[]}
    train_loader = DataLoader(self.train, batch_size=16, shuffle=True)
    val_loader = DataLoader(self.val, batch_size=16, shuffle=True)
    for epoch in tqdm(range(1, epochs + 1)):
      loss_epoch = 0
      self.model.train()

      for x, y in train_loader:
        self.optim.zero_grad()
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        loss.backward()
        self.optim.step()
        loss_epoch += loss.item()
        
      result['loss'].append(loss_epoch / len(train_loader))

      self.model.eval()
      for x, y in val_loader:
        preds = self.model(x)

        for metric in val_metrics:
          metric.update(torch.argmax(preds, dim=-1), y)

      for metric in val_metrics:
        result.get(metric.name, []).append(metric.result())

    return result
