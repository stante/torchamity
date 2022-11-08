import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Learner:
    def __init__(self, train, val, optim, loss_fn, model):
        self.train = train
        self.val = val
        self.optim = optim
        self.loss_fn = loss_fn
        self.model = model

    def fit(self, epochs, metrics=[], val_metrics=[]):
        def _train_batch(x, y):
            self.optim.zero_grad()
            preds = self.model(x)
            loss = self.loss_fn(preds, y)
            loss.backward()
            self.optim.step()

            for metric in metrics:
                metric.update(torch.argmax(preds, dim=-1), y)

            return loss.item()

        def _val_batch(x, y):
            preds = self.model(x)

            for metric in val_metrics:
                metric.update(torch.argmax(preds, dim=-1), y)

        def _prepare_result():
            result = {'loss': []}

            for metric in metrics:
                result.setdefault(metric.name, [])

            for metric in val_metrics:
                result.setdefault(metric.name, [])

            return result

        result = _prepare_result()

        train_loader = DataLoader(self.train, batch_size=16, shuffle=True)
        val_loader = DataLoader(self.val, batch_size=16, shuffle=True)

        pbar = tqdm(range(1, epochs + 1), bar_format="{n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        for epoch in pbar:
            loss_epoch = 0
            self.model.train()

            for x, y in train_loader:
                loss = _train_batch(x, y)
                loss_epoch += loss

            result['loss'].append(loss_epoch / len(train_loader))

            self.model.eval()
            for x, y in val_loader:
                _val_batch(x, y)

            postfix = [f"loss={loss_epoch:.4f}"]
            postfix.extend([f"{metric.name}={metric.result():.4f}" for metric in metrics])
            postfix.extend([f"val_{metric.name}={metric.result():.4f}" for metric in val_metrics])

            pbar.set_postfix_str(', '.join(postfix))

            for metric in val_metrics:
                result[metric.name].append(metric.result())

        return result
