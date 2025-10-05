import torch
import numpy as np
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt

class LayerAnalysis():
    def get_weight_statistics(self, model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
        """
        Returns statistics (mean, std, min, max, abs mean) for each weight tensor in the model.
        """
        stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.data.ndimension() > 1:
                data = param.data.cpu().numpy()
                stats[name] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'abs_mean': float(np.mean(np.abs(data))),
                }
        return stats

    def plot_combined_weight_histograms(self, model: torch.nn.Module, histogram_name: str):
        """
        Plots one combined histogram of all the weights for all the layers in the model.
        """
        combined_weights = torch.empty(0)
        for name, param in model.named_parameters():
            if param.requires_grad and param.data.ndimension() > 1:
                combined_weights = torch.cat((combined_weights, param.data.cpu().flatten()))
                print(name)

        print(combined_weights.shape)
        plt.figure()
        plt.hist(combined_weights.numpy(), bins=100)
        plt.title(histogram_name)
        plt.xlabel('Weight value')
        plt.ylabel('Frequency')
        plt.show()


    def plot_weight_histograms(self, model: torch.nn.Module, histogram_name: str):
        """
        Plots histograms of weights for up to max_layers layers.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and param.data.ndimension() > 1:
                plt.figure()
                plt.hist(param.data.cpu().numpy().flatten(), bins=100)
                plt.title(f'Weight Histogram: {name}')
                plt.xlabel('Weight value')
                plt.ylabel('Frequency')
                # plt.show()

    def compare_model_weights(self, model1: torch.nn.Module, model2: torch.nn.Module) -> Dict[str, Dict[str, float]]:
        """
        Compares weights between two models (e.g., before and after training).
        Returns dict with L1, L2, and cosine similarity for each matching parameter.
        """
        comparison = {}
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())
        for name in params1:
            if name in params2 and params1[name].data.shape == params2[name].data.shape:
                w1 = params1[name].data.cpu().numpy().flatten()
                w2 = params2[name].data.cpu().numpy().flatten()
                l1 = float(np.mean(np.abs(w1 - w2)))
                l2 = float(np.sqrt(np.mean((w1 - w2) ** 2)))
                cos_sim = float(np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-8))
                comparison[name] = {
                    'l1_distance': l1,
                    'l2_distance': l2,
                    'cosine_similarity': cos_sim,
                }
        return comparison

    def get_layerwise_norms(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Returns the L2 norm of the weights for each layer.
        """
        norms = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.data.ndimension() > 1:
                norms[name] = float(torch.linalg.norm(param.data).item())
        return norms

    def plot_layerwise_norms(self, model: torch.nn.Module):
        """
        Plots a bar chart of L2 norms for each layer.
        """
        norms = self.get_layerwise_norms(model)
        names = list(norms.keys())
        values = list(norms.values())
        plt.figure(figsize=(10, 5))
        plt.barh(names, values)
        plt.xlabel('L2 Norm')
        plt.title('Layerwise Weight Norms')
        plt.tight_layout()
        plt.show()

    def get_attention_head_statistics(self, model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
        """
        For Vision Transformers: Returns statistics for attention head weights if present.
        """
        stats = {}
        for name, module in model.named_modules():
            if hasattr(module, 'in_proj_weight'):  # Common in ViT attention layers
                data = module.in_proj_weight.data.cpu().numpy()
                stats[name + '.in_proj_weight'] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'abs_mean': float(np.mean(np.abs(data))),
                }
        return stats

    def compare_layerwise_norms(self, model1: torch.nn.Module, model2: torch.nn.Module) -> Dict[str, Tuple[float, float]]:
        """
        Compares L2 norms of each layer between two models.
        Returns a dict with (norm1, norm2) for each matching layer.
        """
        norms1 = self.get_layerwise_norms(model1)
        norms2 = self.get_layerwise_norms(model2)
        comparison = {}
        for name in norms1:
            if name in norms2:
                comparison[name] = (norms1[name], norms2[name])
        return comparison
    