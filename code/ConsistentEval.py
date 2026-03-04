"""
ConsistentEval.py - Consistency Analysis for Counterfactual Explanation Methods

This script analyzes the consistency of CE methods across different recommender
performance levels by comparing Top-1 only vs Top-1-to-5 aggregated evaluation approaches.

"""

import pandas as pd
import numpy as np
import os
import json
import re
import glob
import argparse
import subprocess
import time
import pickle
from pathlib import Path
from collections import defaultdict
from itertools import combinations

# Statistical and visualization imports
from scipy.stats import spearmanr
from scipy.stats import bootstrap as scipy_bootstrap
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch imports for training
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# Import project modules
from recommenders_architecture import VAE, MLP
from help_functions import recommender_evaluations, sample_indices

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# 1. EXPERIMENT GENERATION MODULE
# ============================================================================

def train_and_save_checkpoints(dataset, model_type, num_checkpoints=4,
                                hr_diff_threshold=0.10, max_epochs=50,
                                output_dir='Neucheckpoints/ConsistencyExp'):
    """
    Train one recommender and save checkpoints based on HR@10 performance difference.

    Saves checkpoints when HR@10 improves by at least hr_diff_threshold (default 10%)
    from the last saved checkpoint, collecting num_checkpoints total checkpoints
    representing different performance levels.

    Args:
        dataset: Dataset name (ML1M, Yahoo, Pinterest)
        model_type: Model type (VAE, MLP, LightGCN)
        num_checkpoints: Number of checkpoints to save (default: 4 for 4 performance levels)
        hr_diff_threshold: Minimum relative HR@10 improvement to save checkpoint (default: 0.10 = 10%)
        max_epochs: Maximum training epochs (default: 50)
        output_dir: Directory to save checkpoints

    Returns:
        List of (checkpoint_path, epoch, hr_score) tuples sorted by performance
    """
    print(f"\n{'='*70}")
    print(f"Training {model_type} on {dataset} with adaptive checkpoint saving")
    print(f"Target: {num_checkpoints} checkpoints with {hr_diff_threshold*100:.0f}% HR@10 improvement")
    print(f"{'='*70}\n")

    # Create output directory
    checkpoint_dir = Path(output_dir) / dataset / model_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup paths and configuration
    DP_DIR = Path("processed_data", dataset)
    export_dir = Path(os.getcwd())
    files_path = Path(export_dir, DP_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'  # Use CPU for consistency

    # Dataset configuration
    num_users_dict = {"ML1M": 6037, "Yahoo": 13797, "Pinterest": 19155}
    num_items_dict = {"ML1M": 3381, "Yahoo": 4604, "Pinterest": 9362}
    output_type_dict = {"VAE": "multiple", "MLP": "single", "LightGCN": "single"}

    num_users = num_users_dict[dataset]
    num_items = num_items_dict[dataset]
    output_type = output_type_dict[model_type]

    # Load data
    print(f"Loading data from {files_path}...")
    train_data = pd.read_csv(files_path / f'train_data_{dataset}.csv', index_col=0)
    test_data = pd.read_csv(files_path / f'test_data_{dataset}.csv', index_col=0)
    static_test_data = pd.read_csv(files_path / f'static_test_data_{dataset}.csv', index_col=0)

    with open(files_path / f'pop_dict_{dataset}.pkl', 'rb') as f:
        pop_dict = pickle.load(f)

    train_array = train_data.to_numpy()
    items_array = np.eye(num_items)
    all_items_tensor = torch.Tensor(items_array).to(device)

    # Prepare test data
    for row in range(static_test_data.shape[0]):
        static_test_data.iloc[row, static_test_data.iloc[row, -2]] = 0
    test_array = static_test_data.iloc[:, :-2].to_numpy()

    pop_array = np.zeros(len(pop_dict))
    for key, value in pop_dict.items():
        pop_array[key] = value

    # Create keyword arguments dictionary
    kw_dict = {
        'device': device,
        'num_items': num_items,
        'pop_array': pop_array,
        'all_items_tensor': all_items_tensor,
        'static_test_data': static_test_data,
        'items_array': items_array,
        'output_type': output_type,
        'recommender_name': model_type
    }

    # Initialize model
    print(f"Initializing {model_type} model...")
    batch_size = 1024
    lr = 5e-3

    if model_type == 'MLP':
        hidden_dim = 512
        model = MLP(hidden_dim, **kw_dict)
    elif model_type == 'VAE':
        hidden_dim = [256, 64]
        VAE_config = {
            "enc_dims": hidden_dim,
            "dropout": 0.5,
            "anneal_cap": 0.2,
            "total_anneal_steps": 200000
        }
        model = VAE(VAE_config, **kw_dict)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with adaptive checkpoint saving
    checkpoints = []
    num_training = train_data.shape[0]
    num_batches = int(np.ceil(num_training / batch_size))
    beta = 1.0

    # Track last saved HR@10 for comparison
    last_saved_hr = 0.0
    checkpoints_saved = 0

    print(f"\nStarting training for up to {max_epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")
    print(f"Will save checkpoint when HR@10 improves by ≥{hr_diff_threshold*100:.0f}%\n")

    for epoch in range(max_epochs):
        model.train()
        train_matrix = sample_indices(train_data.copy(), **kw_dict)
        perm = np.random.permutation(num_training)
        epoch_loss = []

        # Learning rate decay
        if epoch != 0 and epoch % 10 == 0:
            lr = 0.1 * lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Training batches
        for b in range(num_batches):
            optimizer.zero_grad()

            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]

            batch_matrix = torch.FloatTensor(train_matrix[batch_idx, :-2]).to(device)
            batch_pos_idx = train_matrix[batch_idx, -2].astype(int)
            batch_neg_idx = train_matrix[batch_idx, -1].astype(int)

            batch_pos_items = torch.Tensor(items_array[batch_pos_idx]).to(device)
            batch_neg_items = torch.Tensor(items_array[batch_neg_idx]).to(device)

            if model_type == 'MLP':
                pos_output = torch.diagonal(model(batch_matrix, batch_pos_items))
                neg_output = torch.diagonal(model(batch_matrix, batch_neg_items))
                pos_loss = torch.mean((torch.ones_like(pos_output) - pos_output) ** 2)
                neg_loss = torch.mean((neg_output) ** 2)
                batch_loss = pos_loss + beta * neg_loss

            elif model_type == 'VAE':
                pos_output, kl_loss = model(batch_matrix)
                # Use log_softmax for VAE as in the original train_one_epoch method
                ce_loss = -(F.log_softmax(pos_output, 1) * batch_matrix).sum(1).mean()
                # Anneal factor for KL loss (use simple 0.2 or implement annealing)
                anneal = 0.2
                batch_loss = ce_loss + kl_loss * anneal

            batch_loss.backward()
            optimizer.step()
            epoch_loss.append(batch_loss.item())

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            hr_at_10, hr_at_50, hr_at_100, mrr, mpr = recommender_evaluations(model, **kw_dict)

        print(f"Epoch {epoch+1}/{max_epochs} - Loss: {np.mean(epoch_loss):.4f}, "
              f"HR@10: {hr_at_10:.4f}, HR@50: {hr_at_50:.4f}")

        # Determine if we should save this checkpoint
        should_save = False

        if checkpoints_saved == 0:
            # Always save the first checkpoint when HR@10 > 0
            if hr_at_10 > 0.09:
                should_save = True
        elif checkpoints_saved < num_checkpoints:
            # Save if HR@10 improved by at least hr_diff_threshold from last saved
            relative_improvement = (hr_at_10 - last_saved_hr) / last_saved_hr if last_saved_hr > 0 else float('inf')
            absolute_improvement = hr_at_10 - last_saved_hr
            if (relative_improvement >= hr_diff_threshold) and (absolute_improvement >= 0.01):
                should_save = True
                print(f"  → HR@10 improved by {relative_improvement*100:.1f}% from last checkpoint")

        if should_save:
            checkpoint_name = f"{model_type}_{dataset}_perf{checkpoints_saved+1}_epoch{epoch+1}_hr{hr_at_10:.4f}.pt"
            checkpoint_path = checkpoint_dir / checkpoint_name
            torch.save(model.state_dict(), checkpoint_path)
            checkpoints.append((str(checkpoint_path), epoch + 1, hr_at_10))
            last_saved_hr = hr_at_10
            checkpoints_saved += 1
            print(f"  ✓ Saved checkpoint {checkpoints_saved}/{num_checkpoints}: {checkpoint_name}")

        # Stop if we have enough checkpoints
        if checkpoints_saved >= num_checkpoints:
            print(f"\n[INFO] Collected {num_checkpoints} checkpoints. Stopping training early at epoch {epoch+1}.")
            break

    # Sort checkpoints by performance
    checkpoints.sort(key=lambda x: x[2])

    print(f"\n{'='*70}")
    print(f"Training complete! Saved {len(checkpoints)} checkpoints across {num_checkpoints} performance levels:")
    for idx, (path, epoch, hr) in enumerate(checkpoints, 1):
        print(f"  Level {idx}: Epoch {epoch}, HR@10 = {hr:.4f} - {Path(path).name}")
    print(f"{'='*70}\n")

    return checkpoints


def load_existing_checkpoints(checkpoint_dir, dataset, model_type):
    """
    Load existing checkpoint files from a directory instead of training.

    Args:
        checkpoint_dir: Directory containing checkpoints
        dataset: Dataset name (ML1M, Yahoo, Pinterest)
        model_type: Model type (VAE, MLP, LightGCN)

    Returns:
        List of (checkpoint_path, epoch, hr_score) tuples sorted by performance
    """
    print(f"\n{'='*70}")
    print(f"Loading existing checkpoints for {model_type} on {dataset}")
    print(f"{'='*70}\n")

    checkpoint_path = Path(checkpoint_dir) / dataset / model_type

    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint directory not found: {checkpoint_path}")
        return []

    # Try new pattern first: MODEL_DATASET_perfN_epochN_hrX.XXXX.pt
    checkpoint_pattern_new = f"{model_type}_{dataset}_perf*_epoch*_hr*.pt"
    checkpoint_files = sorted(checkpoint_path.glob(checkpoint_pattern_new))

    # Fallback to old pattern: MODEL_DATASET_epochN_hrX.XXXX.pt
    if not checkpoint_files:
        checkpoint_pattern_old = f"{model_type}_{dataset}_epoch*_hr*.pt"
        checkpoint_files = sorted(checkpoint_path.glob(checkpoint_pattern_old))

    if not checkpoint_files:
        print(f"[WARNING] No checkpoint files found")
        print(f"[INFO] Tried patterns: {checkpoint_pattern_new}, {checkpoint_pattern_old}")
        print(f"[INFO] Looking in: {checkpoint_path}")
        return []

    checkpoints = []

    # Parse checkpoint filenames to extract perf level, epoch and HR score
    # New format: MODEL_DATASET_perf1_epoch5_hr0.1234.pt
    pattern_new = re.compile(rf'{model_type}_{dataset}_perf(\d+)_epoch(\d+)_hr([\d.]+)\.pt')
    # Old format: MODEL_DATASET_epoch5_hr0.1234.pt
    pattern_old = re.compile(rf'{model_type}_{dataset}_epoch(\d+)_hr([\d.]+)\.pt')

    for ckpt_file in checkpoint_files:
        match_new = pattern_new.match(ckpt_file.name)
        match_old = pattern_old.match(ckpt_file.name)

        if match_new:
            perf_level = int(match_new.group(1))
            epoch = int(match_new.group(2))
            hr_score = float(match_new.group(3))
            checkpoints.append((str(ckpt_file), epoch, hr_score))
            print(f"  Found: {ckpt_file.name} (Perf Level {perf_level}, Epoch {epoch}, HR@10={hr_score:.4f})")
        elif match_old:
            epoch = int(match_old.group(1))
            hr_score = float(match_old.group(2))
            checkpoints.append((str(ckpt_file), epoch, hr_score))
            print(f"  Found: {ckpt_file.name} (Epoch {epoch}, HR@10={hr_score:.4f})")
        else:
            print(f"  Skipping (name doesn't match pattern): {ckpt_file.name}")

    # Sort by performance (HR score)
    checkpoints.sort(key=lambda x: x[2])

    print(f"\n{'='*70}")
    print(f"Loaded {len(checkpoints)} checkpoints:")
    for path, epoch, hr in checkpoints:
        print(f"  Epoch {epoch}: HR@10 = {hr:.4f} - {Path(path).name}")
    print(f"{'='*70}\n")

    return checkpoints


# ============================================================================
# HELPER FUNCTIONS FOR EVALUATION
# ============================================================================

class Explainer(nn.Module):
    """LXR Explainer architecture"""
    def __init__(self, user_size, item_size, hidden_size, device):
        super(Explainer, self).__init__()
        self.device = device
        self.users_fc = nn.Linear(in_features=user_size, out_features=hidden_size).to(device)
        self.items_fc = nn.Linear(in_features=item_size, out_features=hidden_size).to(device)
        self.bottleneck = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=hidden_size * 2, out_features=hidden_size).to(device),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=user_size).to(device),
            nn.Sigmoid()
        ).to(device)

    def forward(self, user_tensor, item_tensor):
        user_output = self.users_fc(user_tensor.float())
        item_output = self.items_fc(item_tensor.float())
        combined_output = torch.cat((user_output, item_output), dim=-1)
        expl_scores = self.bottleneck(combined_output).to(self.device)
        return expl_scores


def load_lxr_explainer(lxr_path, num_items, lxr_dim, device):
    """Load LXR explainer model"""
    explainer = Explainer(num_items, num_items, lxr_dim, device)
    # Handle both absolute and relative paths
    if Path(lxr_path).is_absolute():
        checkpoint_path = Path(lxr_path)
    else:
        checkpoint_path = Path(os.getcwd()) / lxr_path

    lxr_checkpoint = torch.load(checkpoint_path, map_location=device)
    explainer.load_state_dict(lxr_checkpoint)
    explainer.eval()
    for param in explainer.parameters():
        param.requires_grad = False
    return explainer


def load_recommender_model(checkpoint_path, model_type, num_items, device, kw_dict):
    """Load recommender model from checkpoint"""
    if model_type == 'MLP':
        hidden_dim = 512
        model = MLP(hidden_dim, **kw_dict)
    elif model_type == 'VAE':
        hidden_dim = [256, 64]
        VAE_config = {
            "enc_dims": hidden_dim,
            "dropout": 0.5,
            "anneal_cap": 0.2,
            "total_anneal_steps": 200000
        }
        model = VAE(VAE_config, **kw_dict)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load checkpoint - handle path correctly
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)

    recommender_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(recommender_checkpoint)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_user_recommended_item_K(user_tensor, recommender, k, kw_dict):
    """Get the k-th top recommended item for a user"""
    from help_functions import recommender_run

    all_items_tensor = kw_dict['all_items_tensor']
    num_items = kw_dict['num_items']

    user_res = recommender_run(user_tensor, recommender, all_items_tensor, None, 'vector', **kw_dict)[:num_items]
    user_tensor_items = user_tensor[:num_items]
    user_catalog = torch.ones_like(user_tensor_items) - user_tensor_items
    user_recommendations = torch.mul(user_res, user_catalog)
    values, indices = torch.topk(user_recommendations, k)
    return indices[k - 1].item()


def run_evaluation_for_topk(recommender, explainer, test_array, test_data, items_array,
                            k, xp_size, jaccard_dict, cosine_dict, shap_values,
                            item_to_cluster, lime, kw_dict):
    """
    Run complete evaluation for a specific top-k value.
    Returns dictionary of metrics for each explanation method.
    """
    from help_functions import recommender_run, get_top_k, get_index_in_the_list, get_ndcg
    from lime import get_lime_args, distance_to_proximity
    from collections import defaultdict

    device = kw_dict['device']
    num_items = kw_dict['num_items']

    # Generate explanations for all test users
    print(f"  Generating explanations for {test_array.shape[0]} users...")

    jaccard_expl_dict = {}
    cosine_expl_dict = {}
    lime_expl_dict = {}
    accent_expl_dict = {}
    shap_expl_dict = {}
    lxr_expl_dict = {}

    with torch.no_grad():
        for i in range(test_array.shape[0]):
            if i % 500 == 0:
                print(f"    Processing user {i}/{test_array.shape[0]}...")

            user_vector = test_array[i]
            user_tensor = torch.FloatTensor(user_vector).to(device)
            user_id = int(test_data.index[i])

            item_id = get_user_recommended_item_K(user_tensor, recommender, k, kw_dict)
            item_vector = items_array[item_id]
            item_tensor = torch.FloatTensor(item_vector).to(device)

            user_vector[item_id] = 0
            user_tensor[item_id] = 0

            # Generate explanations using different methods
            jaccard_expl_dict[user_id] = single_user_expl(
                user_vector, user_tensor, item_id, item_tensor, num_items, recommender,
                jaccard_dict, cosine_dict, shap_values, item_to_cluster, explainer, lime,
                kw_dict, mask_type='jaccard', user_id=user_id
            )
            cosine_expl_dict[user_id] = single_user_expl(
                user_vector, user_tensor, item_id, item_tensor, num_items, recommender,
                jaccard_dict, cosine_dict, shap_values, item_to_cluster, explainer, lime,
                kw_dict, mask_type='cosine', user_id=user_id
            )
            lime_expl_dict[user_id] = single_user_expl(
                user_vector, user_tensor, item_id, item_tensor, num_items, recommender,
                jaccard_dict, cosine_dict, shap_values, item_to_cluster, explainer, lime,
                kw_dict, mask_type='lime', user_id=user_id
            )
            accent_expl_dict[user_id] = single_user_expl(
                user_vector, user_tensor, item_id, item_tensor, num_items, recommender,
                jaccard_dict, cosine_dict, shap_values, item_to_cluster, explainer, lime,
                kw_dict, mask_type='accent', user_id=user_id
            )
            shap_expl_dict[user_id] = single_user_expl(
                user_vector, user_tensor, item_id, item_tensor, num_items, recommender,
                jaccard_dict, cosine_dict, shap_values, item_to_cluster, explainer, lime,
                kw_dict, mask_type='shap', user_id=user_id
            )
            lxr_expl_dict[user_id] = single_user_expl(
                user_vector, user_tensor, item_id, item_tensor, num_items, recommender,
                jaccard_dict, cosine_dict, shap_values, item_to_cluster, explainer, lime,
                kw_dict, mask_type='lxr', user_id=user_id
            )

    # Evaluate each explanation method
    print(f"  Evaluating explanation methods...")
    expl_names_list = ['lxr', 'jaccard', 'cosine', 'lime', 'shap', 'accent']
    expl_dicts = [lxr_expl_dict, jaccard_expl_dict, cosine_expl_dict,
                  lime_expl_dict, shap_expl_dict, accent_expl_dict]

    results = {}
    for expl_name, expl_dict in zip(expl_names_list, expl_dicts):
        print(f"    Evaluating {expl_name}...")
        metrics = eval_one_expl_type(
            expl_name, expl_dict, test_array, test_data, items_array,
            recommender, k, xp_size, kw_dict
        )
        results[expl_name] = metrics

    return results


def single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender,
                    jaccard_dict, cosine_dict, shap_values, item_to_cluster, explainer, lime,
                    kw_dict, mask_type=None, user_id=None):
    """Generate explanation for a single user using specified method"""
    from help_functions import recommender_run, get_top_k
    from lime import get_lime_args, distance_to_proximity

    user_hist_size = int(np.sum(user_vector))
    device = kw_dict['device']
    all_items_tensor = kw_dict['all_items_tensor']

    if mask_type == 'lime':
        lime.kernel_fn = distance_to_proximity
        neighborhood_data, neighborhood_labels, distances, item_id_lime = get_lime_args(
            user_vector, item_id, recommender, all_items_tensor,
            min_pert=50, max_pert=100, num_of_perturbations=150, seed=item_id, **kw_dict
        )
        POS_sim_items = lime.explain_instance_with_data(
            neighborhood_data, neighborhood_labels, distances, item_id,
            user_hist_size, 'highest_weights', pos_neg='POS'
        )
    elif mask_type == 'jaccard':
        sim_items = find_jaccard_mask(user_tensor, item_id, jaccard_dict)
        POS_sim_items = list(sorted(sim_items.items(), key=lambda item: item[1], reverse=True))[:user_hist_size]
    elif mask_type == 'cosine':
        sim_items = find_cosine_mask(user_tensor, item_id, cosine_dict)
        POS_sim_items = list(sorted(sim_items.items(), key=lambda item: item[1], reverse=True))[:user_hist_size]
    elif mask_type == 'shap':
        sim_items = find_shapley_mask(user_tensor, user_id, recommender, shap_values, item_to_cluster)
        POS_sim_items = list(sorted(sim_items.items(), key=lambda item: item[1], reverse=True))[:user_hist_size]
    elif mask_type == 'accent':
        sim_items = find_accent_mask(user_tensor, user_id, item_tensor, item_id, recommender, 5, kw_dict)
        POS_sim_items = list(sorted(sim_items.items(), key=lambda item: item[1], reverse=True))[:user_hist_size]
    elif mask_type == 'lxr':
        sim_items = find_lxr_mask(user_tensor, item_tensor, explainer)
        POS_sim_items = list(sorted(sim_items.items(), key=lambda item: item[1], reverse=True))[:user_hist_size]
    else:
        POS_sim_items = []

    return POS_sim_items


def find_jaccard_mask(x, item_id, jaccard_dict):
    """Find Jaccard-based similarity mask"""
    user_hist = x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x
    user_hist[item_id] = 0
    item_jaccard_dict = {}
    for i, j in enumerate(user_hist > 0):
        if j:
            if (i, item_id) in jaccard_dict:
                item_jaccard_dict[i] = jaccard_dict[(i, item_id)]
            else:
                item_jaccard_dict[i] = 0
    return item_jaccard_dict


def find_cosine_mask(x, item_id, cosine_dict):
    """Find cosine-based similarity mask"""
    user_hist = x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x
    user_hist[item_id] = 0
    item_cosine_dict = {}
    for i, j in enumerate(user_hist > 0):
        if j:
            if (i, item_id) in cosine_dict:
                item_cosine_dict[i] = cosine_dict[(i, item_id)]
            else:
                item_cosine_dict[i] = 0
    return item_cosine_dict


def find_shapley_mask(user_tensor, user_id, model, shap_values, item_to_cluster):
    """Find SHAP-based mask"""
    item_shap = {}
    shapley_values = shap_values[shap_values[:, 0].astype(int) == user_id][:, 1:]
    user_vector = user_tensor.cpu().detach().numpy().astype(int)

    for i in np.where(user_vector.astype(int) == 1)[0]:
        items_cluster = item_to_cluster[i]
        item_shap[i] = shapley_values.T[int(items_cluster)][0]

    return item_shap


def find_accent_mask(user_tensor, user_id, item_tensor, item_id, recommender_model, top_k, kw_dict):
    """Find ACCENT-based mask"""
    from help_functions import recommender_run, get_top_k
    from collections import defaultdict

    device = kw_dict['device']
    items_array = kw_dict['items_array']
    num_items = kw_dict['num_items']

    items_accent = defaultdict(float)
    factor = top_k - 1
    user_accent_hist = user_tensor.cpu().detach().numpy().astype(int)

    # Get topk items
    sorted_indices = list(get_top_k(user_tensor, user_tensor, recommender_model, **kw_dict).keys())

    if top_k == 1:
        top_k_indices = [sorted_indices[0]]
    else:
        top_k_indices = sorted_indices[:top_k]

    for iteration, item_k_id in enumerate(top_k_indices):
        user_accent_hist[item_k_id] = 0
        user_tensor_temp = torch.FloatTensor(user_accent_hist).to(device)
        item_vector = items_array[item_k_id]
        item_tensor_temp = torch.FloatTensor(item_vector).to(device)

        # Check influence of items in history on this specific item in topk
        fia_dict = find_fia_mask(user_tensor_temp, item_tensor_temp, item_k_id, recommender_model, kw_dict)

        if not iteration:
            for key in fia_dict.keys():
                items_accent[key] *= factor
        else:
            for key in fia_dict.keys():
                items_accent[key] -= fia_dict[key]

    for key in items_accent.keys():
        items_accent[key] *= -1

    return items_accent


def find_fia_mask(user_tensor, item_tensor, item_id, recommender, kw_dict):
    """Find Feature Influence Analysis mask"""
    from help_functions import recommender_run

    device = kw_dict['device']
    num_items = kw_dict['num_items']

    y_pred = recommender_run(user_tensor, recommender, item_tensor, item_id, **kw_dict).to(device)
    items_fia = {}
    user_hist = user_tensor.cpu().detach().numpy().astype(int).copy()

    for i in range(num_items):
        if user_hist[i] == 1:
            user_hist[i] = 0
            user_tensor_temp = torch.FloatTensor(user_hist).to(device)
            y_pred_without_item = recommender_run(user_tensor_temp, recommender, item_tensor, item_id, **kw_dict).to(device)
            infl_score = y_pred - y_pred_without_item
            items_fia[i] = float(infl_score.detach().cpu().numpy())
            user_hist[i] = 1

    return items_fia


def find_lxr_mask(x, item_tensor, explainer):
    """Find LXR-based mask"""
    user_hist = x
    expl_scores = explainer(user_hist, item_tensor)
    x_masked = user_hist * expl_scores
    item_sim_dict = {}
    for i, j in enumerate(x_masked > 0):
        if j:
            item_sim_dict[i] = float(x_masked[i].detach().cpu().numpy())
    return item_sim_dict


def eval_one_expl_type(expl_name, expl_dict, test_array, test_data, items_array,
                       recommender, k, xp_size, kw_dict):
    """Evaluate one explanation type and return aggregated metrics"""
    from help_functions import recommender_run, get_top_k, get_index_in_the_list, get_ndcg

    device = kw_dict['device']
    num_items = kw_dict['num_items']

    num_of_bins = test_array.shape[0] + 1
    users_DEL = np.zeros(num_of_bins)
    users_INS = np.zeros(num_of_bins)
    NDCG = np.zeros(num_of_bins)
    POS_at_1 = np.zeros(num_of_bins)
    POS_at_5 = np.zeros(num_of_bins)
    POS_at_10 = np.zeros(num_of_bins)
    POS_at_20 = np.zeros(num_of_bins)
    POS_at_50 = np.zeros(num_of_bins)
    POS_at_100 = np.zeros(num_of_bins)
    NEG_at_1 = np.zeros(num_of_bins)
    NEG_at_5 = np.zeros(num_of_bins)
    NEG_at_10 = np.zeros(num_of_bins)
    NEG_at_20 = np.zeros(num_of_bins)
    NEG_at_50 = np.zeros(num_of_bins)
    NEG_at_100 = np.zeros(num_of_bins)

    num_of_bins = 10

    with torch.no_grad():
        for i in range(test_array.shape[0]):
            user_vector = test_array[i]
            user_tensor = torch.FloatTensor(user_vector).to(device)
            user_id = int(test_data.index[i])

            item_id = get_user_recommended_item_K(user_tensor, recommender, k, kw_dict)
            item_vector = items_array[item_id]
            item_tensor = torch.FloatTensor(item_vector).to(device)

            user_vector[item_id] = 0
            user_tensor[item_id] = 0

            user_expl = expl_dict[user_id]

            res = single_user_metrics(user_vector, user_tensor, item_id, item_tensor,
                                     num_of_bins, recommender, user_expl, xp_size, kw_dict)

            users_DEL[i] = res[0]
            users_INS[i] = res[1]
            NDCG[i] = res[2]
            POS_at_1[i] = res[3]
            POS_at_5[i] = res[4]
            POS_at_10[i] = res[5]
            POS_at_20[i] = res[6]
            POS_at_50[i] = res[7]
            POS_at_100[i] = res[8]
            NEG_at_1[i] = res[9]
            NEG_at_5[i] = res[10]
            NEG_at_10[i] = res[11]
            NEG_at_20[i] = res[12]
            NEG_at_50[i] = res[13]
            NEG_at_100[i] = res[14]

    return {
        'DEL': float(np.mean(users_DEL)),
        'INS': float(np.mean(users_INS)),
        'NDCG': float(np.mean(NDCG)),
        'POS_at_1': float(np.mean(POS_at_1)),
        'POS_at_5': float(np.mean(POS_at_5)),
        'POS_at_10': float(np.mean(POS_at_10)),
        'POS_at_20': float(np.mean(POS_at_20)),
        'POS_at_50': float(np.mean(POS_at_50)),
        'POS_at_100': float(np.mean(POS_at_100)),
        'NEG_at_1': float(np.mean(NEG_at_1)),
        'NEG_at_5': float(np.mean(NEG_at_5)),
        'NEG_at_10': float(np.mean(NEG_at_10)),
        'NEG_at_20': float(np.mean(NEG_at_20)),
        'NEG_at_50': float(np.mean(NEG_at_50)),
        'NEG_at_100': float(np.mean(NEG_at_100)),
    }


def single_user_metrics(user_vector, user_tensor, item_id, item_tensor, num_of_bins,
                       recommender, expl_dict, xp_size, kw_dict):
    """Calculate metrics for a single user explanation"""
    from help_functions import recommender_run, get_top_k, get_index_in_the_list, get_ndcg

    device = kw_dict['device']
    num_items = kw_dict['num_items']

    POS_masked = user_tensor.clone()
    NEG_masked = user_tensor.clone()
    POS_masked[item_id] = 0
    NEG_masked[item_id] = 0
    user_hist_size = int(np.sum(user_vector))

    bins = [0] + [len(x) for x in np.array_split(np.arange(user_hist_size), user_hist_size, axis=0)]

    POS_at_1 = [0] * len(bins)
    POS_at_5 = [0] * len(bins)
    POS_at_10 = [0] * len(bins)
    POS_at_20 = [0] * len(bins)
    POS_at_50 = [0] * len(bins)
    POS_at_100 = [0] * len(bins)

    NEG_at_1 = [0] * len(bins)
    NEG_at_5 = [0] * len(bins)
    NEG_at_10 = [0] * len(bins)
    NEG_at_20 = [0] * len(bins)
    NEG_at_50 = [0] * len(bins)
    NEG_at_100 = [0] * len(bins)

    DEL = [0] * len(bins)
    INS = [0] * len(bins)
    NDCG = [0] * len(bins)

    POS_sim_items = expl_dict
    NEG_sim_items = list(sorted(dict(POS_sim_items).items(), key=lambda item: item[1], reverse=False))

    total_items = 0
    for i in range(min(len(bins), xp_size)):
        total_items += bins[i]

        POS_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=device)
        for j in POS_sim_items[:total_items]:
            POS_masked[j[0]] = 1
        POS_masked = user_tensor - POS_masked

        NEG_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=device)
        for j in NEG_sim_items[:total_items]:
            NEG_masked[j[0]] = 1
        NEG_masked = user_tensor - NEG_masked

        POS_ranked_list = get_top_k(POS_masked, user_tensor, recommender, **kw_dict)

        if item_id in list(POS_ranked_list.keys()):
            POS_index = list(POS_ranked_list.keys()).index(item_id) + 1
        else:
            POS_index = num_items

        NEG_index = get_index_in_the_list(NEG_masked, user_tensor, item_id, recommender, **kw_dict) + 1

        POS_at_1[i] = 1 if POS_index <= 1 else 0
        POS_at_5[i] = 1 if POS_index <= 5 else 0
        POS_at_10[i] = 1 if POS_index <= 10 else 0
        POS_at_20[i] = 1 if POS_index <= 20 else 0
        POS_at_50[i] = 1 if POS_index <= 50 else 0
        POS_at_100[i] = 1 if POS_index <= 100 else 0

        NEG_at_1[i] = 1 if NEG_index <= 1 else 0
        NEG_at_5[i] = 1 if NEG_index <= 5 else 0
        NEG_at_10[i] = 1 if NEG_index <= 10 else 0
        NEG_at_20[i] = 1 if NEG_index <= 20 else 0
        NEG_at_50[i] = 1 if NEG_index <= 50 else 0
        NEG_at_100[i] = 1 if NEG_index <= 100 else 0

        DEL[i] = float(recommender_run(POS_masked, recommender, item_tensor, item_id, **kw_dict).detach().cpu().numpy())
        INS[i] = float(recommender_run(user_tensor - POS_masked, recommender, item_tensor, item_id, **kw_dict).detach().cpu().numpy())
        NDCG[i] = get_ndcg(list(POS_ranked_list.keys()), item_id, **kw_dict)

    res = [DEL, INS, NDCG, POS_at_1, POS_at_5, POS_at_10, POS_at_20, POS_at_50, POS_at_100,
           NEG_at_1, NEG_at_5, NEG_at_10, NEG_at_20, NEG_at_50, NEG_at_100]

    for i in range(len(res)):
        res[i] = np.array(res[i])
        res[i] = sum(res[i]) / len(res[i])

    return res


def write_metrics_output(output_file, metrics_results, checkpoint_path, model_type):
    """Write metrics results to output file in the expected format"""
    with open(output_file, 'w') as f:
        f.write(f"this experiment is for {checkpoint_path} and {model_type}\n")

        for expl_name, metrics in metrics_results.items():
            f.write(f"users_DEL_{expl_name}: {metrics['DEL']}\n")
            f.write(f"users_INS_{expl_name}: {metrics['INS']}\n")
            f.write(f"NDCG_{expl_name}: {metrics['NDCG']}\n")
            f.write(f"POS_at_1_{expl_name}: {metrics['POS_at_1']}\n")
            f.write(f"POS_at_5_{expl_name}: {metrics['POS_at_5']}\n")
            f.write(f"POS_at_10_{expl_name}: {metrics['POS_at_10']}\n")
            f.write(f"POS_at_20_{expl_name}: {metrics['POS_at_20']}\n")
            f.write(f"POS_at_50_{expl_name}: {metrics['POS_at_50']}\n")
            f.write(f"POS_at_100_{expl_name}: {metrics['POS_at_100']}\n")
            f.write(f"NEG_at_1_{expl_name}: {metrics['NEG_at_1']}\n")
            f.write(f"NEG_at_5_{expl_name}: {metrics['NEG_at_5']}\n")
            f.write(f"NEG_at_10_{expl_name}: {metrics['NEG_at_10']}\n")
            f.write(f"NEG_at_20_{expl_name}: {metrics['NEG_at_20']}\n")
            f.write(f"NEG_at_50_{expl_name}: {metrics['NEG_at_50']}\n")
            f.write(f"NEG_at_100_{expl_name}: {metrics['NEG_at_100']}\n")


# ============================================================================
# EXPERIMENT GENERATION MODULE (CONTINUED)
# ============================================================================

def run_topk_experiments(checkpoint_paths, lxr_path, dataset, model_type,
                         topk_values=[1, 2, 3, 4, 5], xp_size=10,
                         output_base_dir='Results/ConsistencyAnalysis', test_limit=None):
    """
    Run evaluation experiments for all checkpoints and top-k values.

    This implements the metricsTopK.py logic directly instead of calling it as subprocess.

    Args:
        checkpoint_paths: List of (checkpoint_path, epoch, hr_score) tuples
        lxr_path: Path to LXR explainer checkpoint
        dataset: Dataset name
        model_type: Model type
        topk_values: List of k values to test
        xp_size: Explanation size parameter
        output_base_dir: Base directory for outputs
        test_limit: Limit number of test users (None = use all)

    Returns:
        Dictionary mapping (perf_level, topk) -> output_file_path
    """
    print(f"\n{'='*70}")
    print(f"Running Top-K Experiments")
    print(f"{'='*70}\n")

    # Import baselines
    import sys
    sys.path.append('baselines')
    from lime import LimeBase, distance_to_proximity, get_lime_args
    from help_functions import recommender_run, get_top_k, get_index_in_the_list, get_ndcg

    results_map = {}
    total_experiments = len(checkpoint_paths) * len(topk_values)
    current_exp = 0

    # Setup device and paths
    device = torch.device('cpu')
    files_path = Path(os.getcwd()) / "processed_data" / dataset

    # Load dataset configurations
    num_users_dict = {"ML1M": 6037, "Yahoo": 13797, "Pinterest": 19155}
    num_items_dict = {"ML1M": 3381, "Yahoo": 4604, "Pinterest": 9362}
    output_type_dict = {"VAE": "multiple", "MLP": "single", "LightGCN": "single"}

    num_users = num_users_dict[dataset]
    num_items = num_items_dict[dataset]
    output_type = output_type_dict[model_type]

    # Load data files (once for all experiments)
    print("Loading dataset and preprocessed dictionaries...")
    train_data = pd.read_csv(files_path / f'train_data_{dataset}.csv', index_col=0)
    test_data = pd.read_csv(files_path / f'test_data_{dataset}.csv', index_col=0)
    static_test_data = pd.read_csv(files_path / f'static_test_data_{dataset}.csv', index_col=0)

    with open(files_path / f'pop_dict_{dataset}.pkl', 'rb') as f:
        pop_dict = pickle.load(f)
    with open(files_path / f'jaccard_based_sim_{dataset}.pkl', 'rb') as f:
        jaccard_dict = pickle.load(f)
    with open(files_path / f'cosine_based_sim_{dataset}.pkl', 'rb') as f:
        cosine_dict = pickle.load(f)
    with open(files_path / f'tf_idf_dict_{dataset}.pkl', 'rb') as f:
        tf_idf_dict = pickle.load(f)
    with open(files_path / f'item_to_cluster_{model_type}_{dataset}.pkl', 'rb') as f:
        item_to_cluster = pickle.load(f)
    with open(files_path / f'shap_values_{model_type}_{dataset}.pkl', 'rb') as f:
        shap_values = pickle.load(f)

    # Prepare arrays
    train_array = train_data.to_numpy()
    test_array = static_test_data.iloc[:, :-2].to_numpy()

    # Apply test limit if specified
    if test_limit is not None and test_limit > 0:
        original_size = test_array.shape[0]
        test_array = test_array[:test_limit]
        test_data = test_data.iloc[:test_limit]
        print(f"[DEBUG] Limiting test users: {original_size} -> {test_limit}")

    items_array = np.eye(num_items)
    all_items_tensor = torch.Tensor(items_array).to(device)

    pop_array = np.zeros(len(pop_dict))
    for key, value in pop_dict.items():
        pop_array[key] = value

    # Make similarity dicts symmetric
    for i in range(num_items):
        for j in range(i, num_items):
            jaccard_dict[(j, i)] = jaccard_dict[(i, j)]
            cosine_dict[(j, i)] = cosine_dict[(i, j)]

    # Create keyword arguments
    kw_dict = {
        'device': device,
        'num_items': num_items,
        'pop_array': pop_array,
        'all_items_tensor': all_items_tensor,
        'static_test_data': static_test_data,
        'items_array': items_array,
        'output_type': output_type,
        'recommender_name': model_type
    }

    # Load LXR explainer (once for all experiments)
    print(f"Loading LXR explainer from {lxr_path}...")
    lxr_dim = 128  # Default, could be made configurable
    explainer = load_lxr_explainer(lxr_path, num_items, lxr_dim, device)

    # Initialize LIME
    lime = LimeBase(distance_to_proximity)

    # Run experiments for each checkpoint and top-k value
    for perf_idx, (checkpoint_path, epoch, hr_score) in enumerate(checkpoint_paths, 1):
        # Load recommender model
        print(f"\nLoading recommender from {checkpoint_path}...")
        recommender = load_recommender_model(checkpoint_path, model_type, num_items, device, kw_dict)

        for k in topk_values:
            current_exp += 1
            print(f"\n[{current_exp}/{total_experiments}] Running experiment:")
            print(f"  Performance Level: {perf_idx} (Epoch {epoch}, HR={hr_score:.4f})")
            print(f"  Top-K: {k}")

            # Create output directory
            output_dir = Path(output_base_dir) / dataset / model_type / f"perf{perf_idx}" / f"topk{k}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "metrics.out"

            try:
                # Run evaluation
                metrics_results = run_evaluation_for_topk(
                    recommender, explainer, test_array, test_data, items_array,
                    k, xp_size, jaccard_dict, cosine_dict, shap_values,
                    item_to_cluster, lime, kw_dict
                )

                # Write results to file
                write_metrics_output(output_file, metrics_results, checkpoint_path, model_type)

                print(f"  ✓ Success - saved to {output_file}")
                results_map[(perf_idx, k)] = str(output_file)

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Completed {len(results_map)}/{total_experiments} experiments successfully")
    print(f"{'='*70}\n")

    return results_map


# ============================================================================
# 2. DATA PARSING MODULE
# ============================================================================

def parse_metrics_output(file_path):
    """
    Parse metricsTopK.py output file to extract all metrics.

    Args:
        file_path: Path to output file

    Returns:
        Dictionary with structure: {method: {metric: value}}
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return {}

    results = defaultdict(dict)

    # Regex patterns for different metrics
    patterns = {
        'pos_metric': re.compile(r'POS_at_(\d+)_(\w+):\s+([\d.]+)'),
        'neg_metric': re.compile(r'NEG_at_(\d+)_(\w+):\s+([\d.]+)'),
        'del': re.compile(r'users_DEL_(\w+):\s+([\d.]+)'),
        'ins': re.compile(r'users_INS_(\w+):\s+([\d.]+)'),
        'ndcg': re.compile(r'NDCG_(\w+):\s+([\d.]+)'),
    }

    with open(file_path, 'r') as f:
        content = f.read()

        # Parse POS metrics
        for match in patterns['pos_metric'].finditer(content):
            k, method, value = match.groups()
            results[method][f'POS_at_{k}'] = float(value)

        # Parse NEG metrics
        for match in patterns['neg_metric'].finditer(content):
            k, method, value = match.groups()
            results[method][f'NEG_at_{k}'] = float(value)

        # Parse DEL metrics
        for match in patterns['del'].finditer(content):
            method, value = match.groups()
            results[method]['DEL'] = float(value)

        # Parse INS metrics
        for match in patterns['ins'].finditer(content):
            method, value = match.groups()
            results[method]['INS'] = float(value)

        # Parse NDCG metrics
        for match in patterns['ndcg'].finditer(content):
            method, value = match.groups()
            results[method]['NDCG'] = float(value)

    return dict(results)


def load_all_experiments(output_dir, dataset, model):
    """
    Load all experiment results from directory structure.

    Args:
        output_dir: Base output directory
        dataset: Dataset name
        model: Model type

    Returns:
        Nested dictionary: results[perf_level][topk][method][metric]
    """
    print(f"\n{'='*70}")
    print(f"Loading experiment results from {output_dir}")
    print(f"{'='*70}\n")

    results = defaultdict(lambda: defaultdict(dict))

    base_path = Path(output_dir) / dataset / model

    if not base_path.exists():
        print(f"Error: Directory not found: {base_path}")
        return results

    # Scan for perf*/topk*/ directories
    perf_dirs = sorted(base_path.glob("perf*"))

    for perf_dir in perf_dirs:
        perf_level = int(perf_dir.name.replace('perf', ''))

        topk_dirs = sorted(perf_dir.glob("topk*"))
        for topk_dir in topk_dirs:
            topk = int(topk_dir.name.replace('topk', ''))

            metrics_file = topk_dir / "metrics.out"
            if metrics_file.exists():
                print(f"  Loading: perf{perf_level}/topk{topk}")
                parsed = parse_metrics_output(metrics_file)
                results[perf_level][topk] = parsed
            else:
                print(f"  Warning: Missing {metrics_file}")

    print(f"\nLoaded {len(results)} performance levels")
    for perf_level in sorted(results.keys()):
        print(f"  Performance Level {perf_level}: {len(results[perf_level])} top-k values")

    return dict(results)


# ============================================================================
# 3. AGGREGATION STRATEGIES
# ============================================================================

def aggregate_topk_metrics(results, perf_level, metric_name, topk_range=[1, 2, 3, 4, 5],
                           aggregation='mean'):
    """
    Aggregate metrics across top-k values for a specific performance level.

    Args:
        results: Nested dictionary from load_all_experiments
        perf_level: Performance level to aggregate
        metric_name: Metric to aggregate (e.g., 'POS_at_5')
        topk_range: List of top-k values to aggregate
        aggregation: Aggregation method ('mean', 'weighted_linear', 'weighted_exp')

    Returns:
        Dictionary: {method: aggregated_value}
    """
    if perf_level not in results:
        return {}

    aggregated = {}

    # Get all methods from first available topk
    first_topk = topk_range[0]
    if first_topk not in results[perf_level]:
        return {}

    methods = results[perf_level][first_topk].keys()

    for method in methods:
        values = []
        for k in topk_range:
            if k in results[perf_level] and method in results[perf_level][k]:
                if metric_name in results[perf_level][k][method]:
                    values.append(results[perf_level][k][method][metric_name])

        if not values:
            continue

        # Apply aggregation strategy
        if aggregation == 'mean':
            aggregated[method] = np.mean(values)

        elif aggregation == 'weighted_linear':
            # Weights: 50%, 25%, 15%, 6%, 4% for k=1,2,3,4,5
            weights = np.array([0.5, 0.25, 0.15, 0.06, 0.04])[:len(values)]
            weights = weights / weights.sum()  # Normalize if fewer values
            aggregated[method] = np.average(values, weights=weights)

        elif aggregation == 'weighted_exp':
            # Exponential decay: w_k = exp(-λ*k)
            lambda_param = 0.5
            weights = np.exp(-lambda_param * np.arange(len(values)))
            weights = weights / weights.sum()
            aggregated[method] = np.average(values, weights=weights)

        elif aggregation == 'harmonic':
            # Harmonic mean
            aggregated[method] = len(values) / np.sum(1.0 / np.array(values))

    return aggregated


# ============================================================================
# 4. CONSISTENCY MEASUREMENT MODULE
# ============================================================================

def calculate_method_rankings(metrics_dict, metric_name, higher_is_better=True):
    """
    Calculate ranking of methods for a specific metric.

    Args:
        metrics_dict: Dictionary {method: {metric: value}}
        metric_name: Metric to rank by
        higher_is_better: Whether higher values are better

    Returns:
        Ordered list of method names from best to worst
    """
    method_values = []
    for method, metrics in metrics_dict.items():
        if metric_name in metrics:
            method_values.append((method, metrics[metric_name]))

    # Sort by value
    method_values.sort(key=lambda x: x[1], reverse=higher_is_better)

    return [method for method, _ in method_values]


def calculate_ranking_fluctuation(rankings_across_perf):
    """
    Calculate consistency metrics for rankings across performance levels.

    Args:
        rankings_across_perf: List of rankings, each ranking is a list of method names

    Returns:
        Dictionary with consistency metrics:
            - rank_variance: Average variance of rank positions
            - spearman_corr: Average Spearman correlation
            - stability_index: Rank stability score
    """
    if len(rankings_across_perf) < 2:
        return {'rank_variance': 0, 'spearman_corr': 1.0, 'stability_index': 1.0}

    # Get all unique methods
    all_methods = set()
    for ranking in rankings_across_perf:
        all_methods.update(ranking)
    all_methods = sorted(all_methods)

    # Convert rankings to numeric ranks
    rank_matrix = []
    for ranking in rankings_across_perf:
        ranks = []
        for method in all_methods:
            if method in ranking:
                ranks.append(ranking.index(method) + 1)
            else:
                ranks.append(len(ranking) + 1)  # Worst possible rank
        rank_matrix.append(ranks)

    rank_matrix = np.array(rank_matrix)

    # 1. Rank Variance (lower is better)
    method_variances = np.var(rank_matrix, axis=0)
    avg_variance = np.mean(method_variances)

    # 2. Spearman Correlation (higher is better)
    correlations = []
    for i, j in combinations(range(len(rankings_across_perf)), 2):
        corr, _ = spearmanr(rank_matrix[i], rank_matrix[j])
        if not np.isnan(corr):
            correlations.append(corr)
    avg_spearman = np.mean(correlations) if correlations else 0.0

    # 3. Stability Index (higher is better)
    rank_changes = 0
    max_possible_changes = len(all_methods) * (len(rankings_across_perf) - 1)

    for method_idx in range(len(all_methods)):
        for i in range(len(rankings_across_perf) - 1):
            if rank_matrix[i][method_idx] != rank_matrix[i+1][method_idx]:
                rank_changes += 1

    stability = 1 - (rank_changes / max_possible_changes) if max_possible_changes > 0 else 1.0

    return {
        'rank_variance': float(avg_variance),
        'spearman_corr': float(avg_spearman),
        'stability_index': float(stability)
    }


def compare_consistency(results, metrics_to_analyze, aggregation_methods,
                       topk_range=[1, 2, 3, 4, 5]):
    """
    Compare consistency across progressive Top-K aggregations.

    Compares: Top-1, Top-1-to-2, Top-1-to-3, Top-1-to-4, Top-1-to-5

    Args:
        results: Nested dictionary from load_all_experiments
        metrics_to_analyze: List of metric names
        aggregation_methods: List of aggregation methods to test
        topk_range: Range of top-k values for aggregation (should be [1,2,3,4,5])

    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'='*70}")
    print(f"Comparing Consistency Across Progressive Top-K Aggregations")
    print(f"{'='*70}\n")

    comparison_results = defaultdict(dict)
    perf_levels = sorted(results.keys())

    for metric_name in metrics_to_analyze:
        print(f"\nAnalyzing metric: {metric_name}")

        # Determine if higher is better
        higher_is_better = not metric_name.startswith('DEL')

        # Top-1 only approach (no aggregation)
        print(f"  Calculating Top-1 only rankings...")
        top1_rankings = []
        for perf_level in perf_levels:
            if 1 in results[perf_level]:
                rankings = calculate_method_rankings(
                    results[perf_level][1], metric_name, higher_is_better
                )
                top1_rankings.append(rankings)

        top1_consistency = calculate_ranking_fluctuation(top1_rankings)
        comparison_results[metric_name]['top1'] = {
            'aggregation': 'none',
            'topk_range': [1],
            'consistency': top1_consistency
        }

        # Progressive aggregated approaches: Top-1-to-2, Top-1-to-3, Top-1-to-4, Top-1-to-5
        for max_k in range(2, max(topk_range) + 1):
            current_range = list(range(1, max_k + 1))  # [1,2] or [1,2,3] or [1,2,3,4] or [1,2,3,4,5]

            for agg_method in aggregation_methods:
                print(f"  Calculating Top-1-to-{max_k} ({agg_method}) rankings...")
                agg_rankings = []

                for perf_level in perf_levels:
                    aggregated = aggregate_topk_metrics(
                        results, perf_level, metric_name, current_range, agg_method
                    )
                    rankings = calculate_method_rankings(
                        {m: {metric_name: v} for m, v in aggregated.items()},
                        metric_name, higher_is_better
                    )
                    agg_rankings.append(rankings)

                agg_consistency = calculate_ranking_fluctuation(agg_rankings)
                comparison_results[metric_name][f'top1to{max_k}_{agg_method}'] = {
                    'aggregation': agg_method,
                    'topk_range': current_range,
                    'consistency': agg_consistency
                }

    return dict(comparison_results)


# ============================================================================
# 5. STATISTICAL SIGNIFICANCE MODULE
# ============================================================================

def bootstrap_consistency(rankings_across_perf, consistency_metric='spearman_corr',
                         n_iterations=1000, confidence_level=0.95):
    """
    Calculate bootstrap confidence intervals for consistency metrics.

    Args:
        rankings_across_perf: List of rankings
        consistency_metric: Which consistency metric to bootstrap
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary with mean, ci_lower, ci_upper
    """
    if len(rankings_across_perf) < 2:
        return {'mean': 1.0, 'ci_lower': 1.0, 'ci_upper': 1.0}

    bootstrap_values = []

    for _ in range(n_iterations):
        # Resample performance levels with replacement
        indices = np.random.choice(len(rankings_across_perf),
                                  size=len(rankings_across_perf),
                                  replace=True)
        resampled_rankings = [rankings_across_perf[i] for i in indices]

        # Calculate consistency
        consistency = calculate_ranking_fluctuation(resampled_rankings)
        bootstrap_values.append(consistency[consistency_metric])

    bootstrap_values = np.array(bootstrap_values)

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_values, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_values, (1 - alpha/2) * 100)

    return {
        'mean': float(np.mean(bootstrap_values)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper)
    }


# ============================================================================
# 6. VISUALIZATION MODULE
# ============================================================================

def plot_consistency_comparison(comparison_results, output_path, dataset, model):
    """
    Create bar plot comparing consistency scores across progressive Top-K aggregations.

    Args:
        comparison_results: Results from compare_consistency()
        output_path: Path to save figure
        dataset: Dataset name for title
        model: Model name for title
    """
    print(f"\nCreating consistency comparison plot...")

    metrics = list(comparison_results.keys())
    consistency_types = ['spearman_corr', 'rank_variance', 'stability_index']

    # Helper function to format approach names
    def format_approach_name(approach_name):
        if approach_name == 'top1':
            return 'Top-1'
        # Replace top1to2_mean -> Top1-2: mean, etc.
        approach_name = approach_name.replace('top1to', 'Top1-')
        for sep in ['_mean', '_weighted_linear', '_weighted_exp', '_harmonic']:
            if sep in approach_name:
                parts = approach_name.split(sep[1:])  # Split by 'mean', 'weighted_linear', etc.
                if len(parts) == 2:
                    return f"{parts[0]}: {sep[1:]}"
        return approach_name

    # Prepare data
    data = []
    for metric in metrics:
        for approach_name, approach_data in comparison_results[metric].items():
            consistency = approach_data['consistency']
            formatted_name = format_approach_name(approach_name)
            for cons_type in consistency_types:
                data.append({
                    'Metric': metric,
                    'Approach': formatted_name,
                    'Consistency Type': cons_type.replace('_', ' ').title(),
                    'Score': consistency[cons_type]
                })

    df = pd.DataFrame(data)

    # Create figure with subplots for each consistency type
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Progressive Top-K Aggregation Consistency: {dataset} / {model}',
                 fontsize=14, fontweight='bold')

    for idx, cons_type in enumerate(consistency_types):
        ax = axes[idx]
        cons_type_title = cons_type.replace('_', ' ').title()
        subset = df[df['Consistency Type'] == cons_type_title]

        # Group bar plot
        metrics_unique = subset['Metric'].unique()
        x = np.arange(len(metrics_unique))
        approaches_unique = sorted(subset['Approach'].unique())

        # Adjust bar width based on number of approaches
        n_approaches = len(approaches_unique)
        width = 0.8 / n_approaches

        for i, approach in enumerate(approaches_unique):
            approach_data = subset[subset['Approach'] == approach]
            values = [approach_data[approach_data['Metric'] == m]['Score'].values[0]
                     if len(approach_data[approach_data['Metric'] == m]) > 0 else 0
                     for m in metrics_unique]
            offset = (i - n_approaches/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=approach, alpha=0.8)

        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title(cons_type_title)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_unique, rotation=45, ha='right')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {output_path}")


def plot_ranking_stability_heatmap(results, metric_name, topk_range, output_path,
                                  dataset, model, aggregation='mean'):
    """
    Create side-by-side heatmaps showing ranking stability.

    Args:
        results: Nested dictionary from load_all_experiments
        metric_name: Metric to visualize
        topk_range: Range for aggregation
        output_path: Path to save figure
        dataset: Dataset name
        model: Model name
        aggregation: Aggregation method for right heatmap
    """
    print(f"\nCreating ranking stability heatmap for {metric_name}...")

    perf_levels = sorted(results.keys())
    higher_is_better = not metric_name.startswith('DEL')

    # Get rankings for Top-1 only
    top1_rankings = []
    for perf_level in perf_levels:
        if 1 in results[perf_level]:
            rankings = calculate_method_rankings(
                results[perf_level][1], metric_name, higher_is_better
            )
            top1_rankings.append(rankings)

    # Get rankings for aggregated
    agg_rankings = []
    for perf_level in perf_levels:
        aggregated = aggregate_topk_metrics(
            results, perf_level, metric_name, topk_range, aggregation
        )
        rankings = calculate_method_rankings(
            {m: {metric_name: v} for m, v in aggregated.items()},
            metric_name, higher_is_better
        )
        agg_rankings.append(rankings)

    # Get all methods
    all_methods = sorted(set(m for ranking in top1_rankings + agg_rankings for m in ranking))

    # Create rank matrices
    top1_matrix = []
    agg_matrix = []

    for rankings in top1_rankings:
        row = [rankings.index(m) + 1 if m in rankings else len(rankings) + 1 for m in all_methods]
        top1_matrix.append(row)

    for rankings in agg_rankings:
        row = [rankings.index(m) + 1 if m in rankings else len(rankings) + 1 for m in all_methods]
        agg_matrix.append(row)

    top1_matrix = np.array(top1_matrix).T
    agg_matrix = np.array(agg_matrix).T

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Ranking Stability: {dataset} / {model} - {metric_name}',
                 fontsize=14, fontweight='bold')

    # Plot Top-1 only
    sns.heatmap(top1_matrix, annot=True, fmt='d', cmap='RdYlGn_r',
                xticklabels=[f'Perf{i}' for i in perf_levels],
                yticklabels=all_methods, cbar_kws={'label': 'Rank'},
                ax=ax1, vmin=1, vmax=len(all_methods))
    ax1.set_title('Top-1 Only')
    ax1.set_xlabel('Performance Level')
    ax1.set_ylabel('CE Method')

    # Plot Aggregated
    sns.heatmap(agg_matrix, annot=True, fmt='d', cmap='RdYlGn_r',
                xticklabels=[f'Perf{i}' for i in perf_levels],
                yticklabels=all_methods, cbar_kws={'label': 'Rank'},
                ax=ax2, vmin=1, vmax=len(all_methods))
    ax2.set_title(f'Top-1-to-5 ({aggregation})')
    ax2.set_xlabel('Performance Level')
    ax2.set_ylabel('')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {output_path}")


def plot_ranking_fluctuation_lines(results, metric_name, topk_range, output_path,
                                  dataset, model, aggregation='mean'):
    """
    Create line plot showing rank position changes.

    Args:
        results: Nested dictionary from load_all_experiments
        metric_name: Metric to visualize
        topk_range: Range for aggregation
        output_path: Path to save figure
        dataset: Dataset name
        model: Model name
        aggregation: Aggregation method
    """
    print(f"\nCreating ranking fluctuation line plot for {metric_name}...")

    perf_levels = sorted(results.keys())
    higher_is_better = not metric_name.startswith('DEL')

    # Get rankings
    top1_rankings = []
    agg_rankings = []

    for perf_level in perf_levels:
        # Top-1
        if 1 in results[perf_level]:
            rankings = calculate_method_rankings(
                results[perf_level][1], metric_name, higher_is_better
            )
            top1_rankings.append(rankings)

        # Aggregated
        aggregated = aggregate_topk_metrics(
            results, perf_level, metric_name, topk_range, aggregation
        )
        rankings = calculate_method_rankings(
            {m: {metric_name: v} for m, v in aggregated.items()},
            metric_name, higher_is_better
        )
        agg_rankings.append(rankings)

    # Get all methods
    all_methods = sorted(set(m for ranking in top1_rankings + agg_rankings for m in ranking))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Ranking Fluctuation: {dataset} / {model} - {metric_name}',
                 fontsize=14, fontweight='bold')

    # Plot Top-1
    for method in all_methods:
        ranks = [rankings.index(method) + 1 if method in rankings else len(rankings) + 1
                for rankings in top1_rankings]
        ax1.plot(perf_levels, ranks, marker='o', label=method, linewidth=2)

    ax1.set_xlabel('Performance Level')
    ax1.set_ylabel('Rank Position')
    ax1.set_title('Top-1 Only')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Rank 1 at top

    # Plot Aggregated
    for method in all_methods:
        ranks = [rankings.index(method) + 1 if method in rankings else len(rankings) + 1
                for rankings in agg_rankings]
        ax2.plot(perf_levels, ranks, marker='o', label=method, linewidth=2)

    ax2.set_xlabel('Performance Level')
    ax2.set_ylabel('Rank Position')
    ax2.set_title(f'Top-1-to-5 ({aggregation})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {output_path}")


def plot_progressive_topk_consistency(comparison_results, output_path, dataset, model, metric_name):
    """
    Create line plot showing how consistency changes with progressive Top-K aggregation.

    Args:
        comparison_results: Results from compare_consistency()
        output_path: Path to save figure
        dataset: Dataset name
        model: Model name
        metric_name: Specific metric to visualize
    """
    print(f"\nCreating progressive Top-K consistency plot for {metric_name}...")

    if metric_name not in comparison_results:
        print(f"  Warning: Metric {metric_name} not found in results")
        return

    metric_results = comparison_results[metric_name]
    consistency_types = ['spearman_corr', 'rank_variance', 'stability_index']

    # Extract data for each aggregation method
    aggregation_methods = set()
    for approach_name, approach_data in metric_results.items():
        if approach_name != 'top1':
            aggregation_methods.add(approach_data['aggregation'])

    # Organize data by aggregation method and topk range
    data_by_agg = {agg: {cons_type: {1: None} for cons_type in consistency_types}
                   for agg in aggregation_methods}

    # Add baseline (Top-1)
    top1_consistency = metric_results['top1']['consistency']
    for agg in aggregation_methods:
        for cons_type in consistency_types:
            data_by_agg[agg][cons_type][1] = top1_consistency[cons_type]

    # Add progressive aggregations
    for approach_name, approach_data in metric_results.items():
        if approach_name == 'top1':
            continue

        topk_range = approach_data.get('topk_range', [])
        max_k = max(topk_range) if topk_range else 1
        agg = approach_data['aggregation']
        consistency = approach_data['consistency']

        for cons_type in consistency_types:
            data_by_agg[agg][cons_type][max_k] = consistency[cons_type]

    # Create figure with 3 subplots (one for each consistency type)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Progressive Top-K Aggregation Effect: {dataset}/{model} - {metric_name}',
                 fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(range(len(aggregation_methods)))
    markers = ['o', 's', '^', 'D', 'v']

    for idx, cons_type in enumerate(consistency_types):
        ax = axes[idx]

        for i, agg in enumerate(sorted(aggregation_methods)):
            # Get sorted k values and corresponding scores
            k_values = sorted(data_by_agg[agg][cons_type].keys())
            scores = [data_by_agg[agg][cons_type][k] for k in k_values]

            # Filter out None values
            valid_pairs = [(k, s) for k, s in zip(k_values, scores) if s is not None]
            if valid_pairs:
                k_values, scores = zip(*valid_pairs)
                ax.plot(k_values, scores, marker=markers[i % len(markers)],
                       label=agg, linewidth=2, markersize=8, color=colors[i])

        ax.set_xlabel('Top-K Range (1 to K)', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(cons_type.replace('_', ' ').title(), fontsize=12)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['Top-1', 'Top-1-2', 'Top-1-3', 'Top-1-4', 'Top-1-5'])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {output_path}")


# ============================================================================
# 7. OUTPUT AND REPORTING
# ============================================================================

def save_results_to_csv(comparison_results, output_path, dataset, model):
    """
    Save consistency scores to CSV file for all progressive Top-K aggregations.

    Args:
        comparison_results: Results from compare_consistency()
        output_path: Path to save CSV
        dataset: Dataset name
        model: Model name
    """
    print(f"\nSaving consistency scores to CSV...")

    rows = []
    for metric, approaches in comparison_results.items():
        top1_consistency = approaches['top1']['consistency']

        for approach_name, approach_data in approaches.items():
            consistency = approach_data['consistency']

            # Extract topk_range for better tracking
            topk_range_str = str(approach_data.get('topk_range', [1]))

            # Calculate improvement over top1
            improvement_pct = 0.0
            improvement_variance = 0.0
            improvement_stability = 0.0

            if approach_name != 'top1':
                # Spearman correlation improvement
                top1_score = top1_consistency['spearman_corr']
                current_score = consistency['spearman_corr']
                if top1_score > 0:
                    improvement_pct = ((current_score - top1_score) / top1_score) * 100

                # Rank variance improvement (lower is better, so negative change is improvement)
                top1_var = top1_consistency['rank_variance']
                current_var = consistency['rank_variance']
                if top1_var > 0:
                    improvement_variance = ((top1_var - current_var) / top1_var) * 100

                # Stability index improvement
                top1_stab = top1_consistency['stability_index']
                current_stab = consistency['stability_index']
                if top1_stab > 0:
                    improvement_stability = ((current_stab - top1_stab) / top1_stab) * 100

            rows.append({
                'dataset': dataset,
                'model': model,
                'metric': metric,
                'approach': approach_name,
                'aggregation': approach_data['aggregation'],
                'topk_range': topk_range_str,
                'spearman_corr': consistency['spearman_corr'],
                'rank_variance': consistency['rank_variance'],
                'stability_index': consistency['stability_index'],
                'improvement_spearman_pct': improvement_pct,
                'improvement_variance_pct': improvement_variance,
                'improvement_stability_pct': improvement_stability
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    return df


def save_rankings_to_csv(results, metric_name, topk_range, output_path,
                         dataset, model, aggregation_methods):
    """
    Save method rankings across performance levels to CSV.

    Args:
        results: Nested dictionary from load_all_experiments
        metric_name: Metric to analyze
        topk_range: Range for aggregation
        output_path: Path to save CSV
        dataset: Dataset name
        model: Model name
        aggregation_methods: List of aggregation methods
    """
    print(f"\nSaving rankings to CSV...")

    perf_levels = sorted(results.keys())
    higher_is_better = not metric_name.startswith('DEL')

    rows = []

    # Top-1 only
    for perf_level in perf_levels:
        if 1 in results[perf_level]:
            rankings = calculate_method_rankings(
                results[perf_level][1], metric_name, higher_is_better
            )
            row = {
                'dataset': dataset,
                'model': model,
                'perf_level': perf_level,
                'metric': metric_name,
                'approach': 'top1',
                'aggregation': 'none'
            }
            for i, method in enumerate(rankings[:6], 1):  # Top 6 methods
                row[f'rank{i}_method'] = method
            rows.append(row)

    # Progressive aggregated approaches: Top-1-to-2, Top-1-to-3, Top-1-to-4, Top-1-to-5
    for max_k in range(2, max(topk_range) + 1):
        current_range = list(range(1, max_k + 1))

        for agg_method in aggregation_methods:
            for perf_level in perf_levels:
                aggregated = aggregate_topk_metrics(
                    results, perf_level, metric_name, current_range, agg_method
                )
                rankings = calculate_method_rankings(
                    {m: {metric_name: v} for m, v in aggregated.items()},
                    metric_name, higher_is_better
                )
                row = {
                    'dataset': dataset,
                    'model': model,
                    'perf_level': perf_level,
                    'metric': metric_name,
                    'approach': f'top1to{max_k}',
                    'aggregation': agg_method,
                    'topk_range': str(current_range)
                }
                for i, method in enumerate(rankings[:6], 1):
                    row[f'rank{i}_method'] = method
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    return df


def print_summary_report(comparison_results, dataset, model):
    """
    Print comprehensive summary report to console for progressive Top-K aggregations.

    Args:
        comparison_results: Results from compare_consistency()
        dataset: Dataset name
        model: Model name
    """
    print(f"\n{'='*70}")
    print(f"PROGRESSIVE TOP-K AGGREGATION ANALYSIS: {dataset} / {model}")
    print(f"{'='*70}\n")

    for metric in comparison_results:
        print(f"\n{'='*70}")
        print(f"Metric: {metric}")
        print(f"{'='*70}\n")

        top1_data = comparison_results[metric]['top1']
        top1_consistency = top1_data['consistency']

        print(f"Top-1 Only (Baseline):")
        print(f"  Spearman ρ:        {top1_consistency['spearman_corr']:.4f}")
        print(f"  Rank Variance:     {top1_consistency['rank_variance']:.4f}")
        print(f"  Stability Index:   {top1_consistency['stability_index']:.4f}")

        # Group approaches by topk range
        approaches_by_k = {}
        for approach_name, approach_data in comparison_results[metric].items():
            if approach_name == 'top1':
                continue

            topk_range = approach_data.get('topk_range', [])
            max_k = max(topk_range) if topk_range else 1

            if max_k not in approaches_by_k:
                approaches_by_k[max_k] = []
            approaches_by_k[max_k].append((approach_name, approach_data))

        # Print each aggregation level
        for max_k in sorted(approaches_by_k.keys()):
            print(f"\n--- Top-1-to-{max_k} Aggregations ---")

            for approach_name, approach_data in approaches_by_k[max_k]:
                consistency = approach_data['consistency']
                agg_name = approach_data['aggregation']

                # Calculate improvements over baseline
                improvement_spearman = ((consistency['spearman_corr'] - top1_consistency['spearman_corr']) /
                                       top1_consistency['spearman_corr'] * 100) if top1_consistency['spearman_corr'] > 0 else 0

                improvement_variance = ((top1_consistency['rank_variance'] - consistency['rank_variance']) /
                                       top1_consistency['rank_variance'] * 100) if top1_consistency['rank_variance'] > 0 else 0

                improvement_stability = ((consistency['stability_index'] - top1_consistency['stability_index']) /
                                        top1_consistency['stability_index'] * 100) if top1_consistency['stability_index'] > 0 else 0

                print(f"\n  Aggregation: {agg_name}")
                print(f"    Spearman ρ:        {consistency['spearman_corr']:.4f} ({improvement_spearman:+.1f}%)")
                print(f"    Rank Variance:     {consistency['rank_variance']:.4f} ({improvement_variance:+.1f}% reduction)")
                print(f"    Stability Index:   {consistency['stability_index']:.4f} ({improvement_stability:+.1f}%)")

    print(f"\n{'='*70}\n")


# ============================================================================
# 8. MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Consistency Analysis for Counterfactual Explanations'
    )

    # Dataset and model arguments
    parser.add_argument('--dataset', default='ML1M',
                       choices=['ML1M', 'Yahoo', 'Pinterest'],
                       help='Dataset name')
    parser.add_argument('--model', default='VAE',
                       choices=['VAE', 'MLP', 'LightGCN'],
                       help='Model type')

    # Experiment configuration
    parser.add_argument('--metrics', nargs='+',
                       default=['POS_at_1', 'POS_at_5', 'POS_at_10', 'NEG_at_1', 'NEG_at_5', 'NEG_at_10', 'NDCG', 'DEL', 'INS'],
                       help='Metrics to analyze')
    parser.add_argument('--topk_values', nargs='+', type=int,
                       default=[1, 2, 3, 4, 5],
                       help='Top-K values to test')
    parser.add_argument('--xp_size', type=int, default=10,
                       help='Explanation size')

    # Training parameters (for adaptive checkpoint saving)
    parser.add_argument('--num_checkpoints', type=int, default=4,
                       help='Number of checkpoints to save (performance levels)')
    parser.add_argument('--hr_diff_threshold', type=float, default=0.10,
                       help='Minimum HR@10 relative improvement to save checkpoint (default: 0.10 = 10%%)')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Maximum training epochs')

    # Aggregation and consistency
    parser.add_argument('--aggregations', nargs='+',
                       default=['mean', 'weighted_linear', 'weighted_exp'],
                       help='Aggregation methods')

    # Directories
    parser.add_argument('--output_dir', default='Results/ConsistencyAnalysis/',
                       help='Output directory for results')
    parser.add_argument('--checkpoint_dir', default='Neucheckpoints/ConsistencyExp',
                       help='Directory for model checkpoints')
    parser.add_argument('--lxr_path', default='',
                       help='Path to LXR explainer checkpoint')

    # Mode
    parser.add_argument('--mode', choices=['generate', 'analyze', 'both'],
                       default='analyze',
                       help='Execution mode')

    # Skip training option
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and load existing checkpoints from checkpoint_dir')

    # Test limit for debugging
    parser.add_argument('--test_limit', type=int, default=None,
                       help='Limit number of test users for debugging (default: use all test users)')

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = Path('Results/Figures')
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # MODE: GENERATE
    # ========================================================================
    if args.mode in ['generate', 'both']:
        print("\n" + "="*70)
        print("MODE: GENERATE - Running Experiments")
        print("="*70 + "\n")

        # Step 1: Get checkpoints (either train new ones or load existing)
        if args.skip_training:
            print("[Step 1/2] Loading existing checkpoints (--skip_training enabled)...")
            checkpoints = load_existing_checkpoints(
                args.checkpoint_dir, args.dataset, args.model
            )
        else:
            print("[Step 1/2] Training recommender and saving checkpoints...")
            checkpoints = train_and_save_checkpoints(
                args.dataset, args.model, args.num_checkpoints,
                args.hr_diff_threshold, args.max_epochs, args.checkpoint_dir
            )

        if not checkpoints:
            print("\n[ERROR] No checkpoints available.")
            if args.skip_training:
                print(f"[INFO] No existing checkpoints found in: {args.checkpoint_dir}/{args.dataset}/{args.model}/")
                print(f"[INFO] Expected filename formats:")
                print(f"       New: {args.model}_{args.dataset}_perf<N>_epoch<E>_hr<X.XXXX>.pt")
                print(f"       Old: {args.model}_{args.dataset}_epoch<E>_hr<X.XXXX>.pt")
            else:
                print("[INFO] Training did not produce enough checkpoints.")
                print(f"[INFO] Target: {args.num_checkpoints} checkpoints with {args.hr_diff_threshold*100:.0f}% HR@10 improvement")
                print(f"[INFO] Try: Increase --max_epochs or decrease --hr_diff_threshold")
            return

        # Validate we have the expected number of checkpoints
        if len(checkpoints) != args.num_checkpoints:
            print(f"\n[WARNING] Expected {args.num_checkpoints} checkpoints but found {len(checkpoints)}")
            if len(checkpoints) < args.num_checkpoints:
                print(f"[INFO] Proceeding with {len(checkpoints)} performance levels instead of {args.num_checkpoints}")
            else:
                print(f"[INFO] Using the first {args.num_checkpoints} checkpoints (by HR@10 score)")
                checkpoints = checkpoints[:args.num_checkpoints]

        # Step 2: Run experiments
        print("\n[Step 2/2] Running Top-K experiments...")
        if not args.lxr_path:
            print("[ERROR] LXR path not specified. Use --lxr_path argument.")
            return

        results_map = run_topk_experiments(
            checkpoints, args.lxr_path, args.dataset, args.model,
            args.topk_values, args.xp_size, args.output_dir, args.test_limit
        )

        print(f"\n[SUCCESS] Generated {len(results_map)} experiment results")

    # ========================================================================
    # MODE: ANALYZE
    # ========================================================================
    if args.mode in ['analyze', 'both']:
        print("\n" + "="*70)
        print("MODE: ANALYZE - Analyzing Consistency")
        print("="*70 + "\n")

        # Step 1: Load experiment results
        print("[Step 1/6] Loading experiment results...")
        results = load_all_experiments(args.output_dir, args.dataset, args.model)

        if not results:
            print("[ERROR] No experiment results found.")
            print(f"[INFO] Expected directory: {args.output_dir}/{args.dataset}/{args.model}/")
            print("[INFO] Run with --mode=generate first, or check directory structure.")
            return

        # Step 2: Compare consistency
        print("\n[Step 2/6] Comparing consistency across approaches...")
        comparison_results = compare_consistency(
            results, args.metrics, args.aggregations, args.topk_values
        )

        # Step 3: Save results to CSV
        print("\n[Step 3/6] Saving results...")
        csv_path = output_dir / f"{args.dataset}_{args.model}_consistency_scores.csv"
        save_results_to_csv(comparison_results, csv_path, args.dataset, args.model)

        # Save rankings
        for metric in args.metrics:
            rankings_path = output_dir / f"{args.dataset}_{args.model}_{metric}_rankings.csv"
            save_rankings_to_csv(
                results, metric, args.topk_values, rankings_path,
                args.dataset, args.model, args.aggregations
            )

        # Step 4: Generate visualizations
        print("\n[Step 4/6] Generating visualizations...")

        # Comparison bar plot (all progressive aggregations)
        comparison_plot_path = figures_dir / f"consistency_comparison_{args.dataset}_{args.model}.png"
        plot_consistency_comparison(
            comparison_results, comparison_plot_path, args.dataset, args.model
        )

        # Progressive Top-K consistency line plots (one for each metric)
        for metric in args.metrics:
            progressive_path = figures_dir / f"progressive_topk_{args.dataset}_{args.model}_{metric}.png"
            plot_progressive_topk_consistency(
                comparison_results, progressive_path, args.dataset, args.model, metric
            )

        # Heatmaps and line plots for each metric
        for metric in args.metrics:
            for agg in args.aggregations:
                # Heatmap
                heatmap_path = figures_dir / f"ranking_stability_{args.dataset}_{args.model}_{metric}_{agg}.png"
                plot_ranking_stability_heatmap(
                    results, metric, args.topk_values, heatmap_path,
                    args.dataset, args.model, agg
                )

                # Line plot
                line_path = figures_dir / f"ranking_fluctuation_{args.dataset}_{args.model}_{metric}_{agg}.png"
                plot_ranking_fluctuation_lines(
                    results, metric, args.topk_values, line_path,
                    args.dataset, args.model, agg
                )

        # Step 5: Print summary
        print("\n[Step 5/6] Generating summary report...")
        print_summary_report(comparison_results, args.dataset, args.model)

        # Step 6: Save JSON summary
        print("[Step 6/6] Saving JSON summary...")
        json_path = output_dir / f"{args.dataset}_{args.model}_summary.json"
        summary = {
            'dataset': args.dataset,
            'model': args.model,
            'metrics_analyzed': args.metrics,
            'aggregation_methods': args.aggregations,
            'topk_range': args.topk_values,
            'results': comparison_results
        }
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved to: {json_path}")

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70 + "\n")


if __name__ == '__main__':
    main()

