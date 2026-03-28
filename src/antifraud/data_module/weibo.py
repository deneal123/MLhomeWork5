"""
Weibo dataset loader for Task3
"""
from antifraud.data_module.base import DatasetLoader


class WeiboLoader(DatasetLoader):
    def load(self, data_root):
        data_dir = data_root / 'weibo_data'
        data_dir.mkdir(exist_ok=True)

        try:
            import torch
            try:
                from torch_geometric.data.storage import GlobalStorage
                torch.serialization.add_safe_globals([GlobalStorage])
            except Exception:
                pass
 
            try:
                from pygod.datasets import load_weibo
                return load_weibo(root=str(data_dir))
            except Exception as e:
                self.logger = getattr(self, 'logger', None)
                if self.logger is not None:
                    self.logger.warning(f'pygod.datasets.load_weibo failed: {e}')

            try:
                from pygod.utils import load_data
                from inspect import signature

                sig = signature(load_data)
                options = []
                if 'root' in sig.parameters:
                    options.append({'root': str(data_dir)})
                if 'root_dir' in sig.parameters:
                    options.append({'root_dir': str(data_dir)})
                if 'data_dir' in sig.parameters:
                    options.append({'data_dir': str(data_dir)})
                options.append({})

                for opts in options:
                    try:
                        return load_data('weibo', **opts)
                    except Exception:
                        continue
                raise RuntimeError('pygod load_data weibo failed for all parameter variations')
            except Exception as e:
                raise RuntimeError(f'Weibo download failed: {e}')
        except Exception as e:
            raise RuntimeError(f"Weibo loading error: {e}")
