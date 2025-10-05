import torch

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False,model='_',dataset='_',cuda_index=1):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self.path='checkpoint/model_'+model+dataset+str(cuda_index)+'.pt'
        self.alt_path='checkpoint/altmodel_'+model+dataset+str(cuda_index)+'.pt'
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False
    
    def init(self,model):
        torch.save(model.state_dict(),self.path)

    def set_alt_path(self):
        self.path=self.alt_path

    def step(self, metrics,model):
        if self.best is None:
            self.best = metrics
            torch.save(model.state_dict(),self.path)
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            #print('Better model')
            self.num_bad_epochs = 0
            self.best = metrics
            torch.save(model.state_dict(),self.path)
            #self.model=model
        else:
            #print('worse model')
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
    
    def get_model(self,model):
        model_path=self.path
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
