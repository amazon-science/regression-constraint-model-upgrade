import torch

class ModelAnalyzer():
    
    def __init__(self, model_info):
        if type(model_info) == str:
            model_info = torch.load(model_info)
        self.pred = model_info['pred'].cpu()
        self.gt = model_info['gt'].cpu()
        self.outputs = model_info['outputs'].cpu() if 'outputs' in model_info else None
        self.features = model_info['features'].cpu() if ('features' in model_info and len(model_info['features']) > 0) else None
        self.ensemble_level = 1

    def NFR(self, old_model):
        return float(((old_model.pred == self.gt) & (self.pred != self.gt)).sum())/len(self.gt)

    def PFR(self, old_model):
        return float(((old_model.pred != self.gt) & (self.pred == self.gt)).sum())/len(self.gt)

    def NFR_frac(self, old_model):
        return float(((self.nf_samples(old_model)) * (self.max_confidence() - self.gt_confidence())).sum()) / len(self.gt)

    def PFR_frac(self, old_model):
        return float(((self.pf_samples(old_model)) * (old_model.max_confidence() - old_model.gt_confidence())).sum()) / len(self.gt)

    def Acc(self):
        return (self.pred == self.gt).sum()*1.0/len(self.gt)

    def max_confidence(self):
        confidence = torch.nn.Softmax(dim=1)(self.outputs)
        max_confidence = confidence.max(dim=1)[0]
        return max_confidence

    def gt_confidence(self):
        confidence = torch.nn.Softmax(dim=1)(self.outputs)
        gt_confidence = [confidence[i, g] for i, g in enumerate(self.gt)]
        gt_confidence = torch.tensor(gt_confidence)
        return gt_confidence

    def gt_confidence_diff(self, old_model):
        return old_model.gt_confidence() - self.gt_confidence()

    def nf_samples_type_I(self, old_model, thresh=0.1):
        return (old_model.pred == self.gt) & (self.pred != self.gt) & (abs(old_model.gt_confidence() - self.gt_confidence()) <= thresh)

    def nf_samples_type_II(self, old_model, thresh=0.5):
        return (old_model.pred == self.gt) & (self.pred != self.gt) & (abs(old_model.gt_confidence() - self.gt_confidence()) >= thresh)

    def nf_samples(self, old_model):
        return (old_model.pred == self.gt) & (self.pred != self.gt)

    def pf_samples(self, old_model):
        return (old_model.pred != self.gt) & (self.pred == self.gt)

    def correct_samples(self):
        return self.pred == self.gt
    
    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        if other == 0:
            return self

        outputs = (self.outputs * self.ensemble_level + other.outputs * other.ensemble_level)/(self.ensemble_level + other.ensemble_level)
        gt = self.gt
        pred = outputs.max(1)[1]
        ens_model = ModelAnalyzer(dict(outputs=outputs, pred=pred, gt=gt))
        ens_model.ensemble_level = self.ensemble_level + other.ensemble_level
        return ens_model

    def dump(self, name):
        model_info = {'pred': self.pred, 'gt': self.gt}
        if self.outputs is not None:
            model_info['outputs'] = self.outputs 
        torch.save(model_info, name)
