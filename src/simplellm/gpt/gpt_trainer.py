class GPTTrainer(object):
    def __init__(self, model, dataloader, epochs, optimizer, loss_fn, max_iter = -1, report_every = 100) -> None:
        self.model = model
        self.dataloader = dataloader
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.max_iter = max_iter
        self.report_every = report_every
    def run(self):
        self.model.train()
        for e in range(self.epochs):
            for i, d in enumerate(self.dataloader):
                data, target = d
                res = self.model(data)
                loss = self.loss_fn(res, target)
                
                if i % self.report_every == 0:
                    print(f"iteration: {i}, epoch: {e}, loss: {loss.item()}")
                loss.backward()
                self.optimizer.step()
                if i == self.max_iter:
                    break

