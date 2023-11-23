from collections import defaultdict


class BaseTrainCfg:
    def __init__(self, *, num_experiments):
        self.num_experiments = num_experiments

        self._models     = None
        self._cofigs     = None
        self._loaders    = None
        self._names      = None
        self._device     = None

        self._def_config_keys = ["model_mod"]

    @property
    def models(self):
        return self._models
	
    @models.setter
    def models(self, new_models):
        assert len(new_models.keys()) == self.num_experiments

        self._models = new_models

    @models.deleter
    def models(self):
        del self._models

    def default_config_key(self, key):
        if key in self._def_config_keys:
            return None
        
        raise KeyError(key)

    @property
    def configs(self):
        return self._configs
	
    @configs.setter
    def configs(self, new_configs):
        assert len(new_configs.keys()) == self.num_experiments

        self._configs = {
            k: defaultdict(lambda: None, new_configs[k])
            for k in new_configs.keys()
        }

    @configs.deleter
    def configs(self):
        del self._configs

    @property
    def loaders (self):
        return self._loaders 
	
    @loaders .setter
    def loaders (self, new_loaders):
        assert len(new_loaders.keys()) == self.num_experiments

        self._loaders  = new_loaders 

    @loaders .deleter
    def loaders (self):
        del self._loaders 

    @property
    def names(self):
        return self._names
	
    @names.setter
    def names(self, new_names):
        assert len(new_names.keys()) == self.num_experiments

        self._names = new_names

    @names.deleter
    def names(self):
        del self._names

    @property
    def root(self):
        return self._root
	
    @root.setter
    def root(self, new_root):
        self._root = new_root

    @root.deleter
    def root(self):
        del self._root

    @property
    def device(self):
        return self._device
	
    @device.setter
    def device(self, new_device):
        self._device = new_device

    @device.deleter
    def device(self):
        del self._device