import os
import torch

class Checkpoint:
    
    ###########################################################################
    def __init__(
            self,
            folder : str,
            nets: dict,
            optimizers: dict,
            measurements
        ):
        
        self.folder = folder
        self.nets = nets
        self.optimizers = optimizers
        self.measurements = measurements
        self.keep_models = set()
        
        # assertions
        for key in nets:
            if key not in optimizers:
                raise RuntimeError(f"'{key}' not in optimizers dict.")
            if not hasattr(nets[key], "to_checkpoint"):
                raise RuntimeError(f"net '{key}' is not implementing to_checkpoint method.")
    
    ###########################################################################
    def save(self, instances, only_model=False):
        if self.measurements is None:
            raise RuntimeError("Measurements not set.")
            
        path_model = os.path.join(
            self.folder, "{}.model"
        )
        path_measures = os.path.join(
            self.folder, "{}.measures"
        )

        # Checkpoint model
        
        save = {
            "instances": instances,
            "nets": {key: item.to_checkpoint() for key, item in self.nets.items()},
            "optimizers": {key: item.state_dict() for key, item in self.optimizers.items()}
        }
        
        torch.save(
            save,
            path_model.format(instances)
        )
        
        
        # Checkpoint measures
        if not only_model:
            m_i, m_other = self.measurements.get_measurements(instances)   

            torch.save(
                m_i,
                path_measures.format("instances")
            )
                
            
            torch.save(
                m_other,
                path_measures.format("other")
            )
            
    ###########################################################################
    @staticmethod
    def load(
            path : str,
            instances : int,
            device : str,
            measurements = None
        ):
        
        checkpoint = torch.load(os.path.join(path, f"{instances}.model"), map_location=device)
        
        if "nets" in checkpoint: # new checkpointing scheme
            nets = {}
            optimizers = {}
            
            for name, net_chkpt in checkpoint["nets"].items():
                nets[name].load(net_chkpt["state"])
                optimizers[name].load_state_dict(checkpoint["optimizers"][name])
            
        del checkpoint
        
        return Checkpoint(path, nets, optimizers, measurements)

