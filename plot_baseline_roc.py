import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

matplotlib.rcParams.update({'font.size': 14})

def parse_filename(filename):
    """
    Parses filename to extract dataset, model, and attack name.
    Filename format: {dataset_name}_{model_name}_{attack_name}.npz
    """
    name = os.path.splitext(filename)[0]
    parts = name.split('_')
    
    # Heuristic:
    # Attack name is the last part.
    # Model name is the second to last part.
    # Dataset name is everything before the model name.
    
    if len(parts) < 3:
        raise ValueError(f"Filename {filename} does not match expected format.")

    attack = parts[-1]
    model = parts[-2]
    dataset = "_".join(parts[:-2])
    
    return dataset, model, attack

def plot_all_rocs(baseline_dir="baseline", output_dir="results/baseline_plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(baseline_dir):
        print(f"Error: Directory '{baseline_dir}' does not exist.")
        return

    files = [f for f in os.listdir(baseline_dir) if f.endswith('.npz')]
    
    # Grouping
    groups = {} # Key: (dataset, model), Value: list of (attack, filepath)
    
    print(f"Found {len(files)} files in {baseline_dir}")

    for f in files:
        try:
            dataset, model, attack = parse_filename(f)
            key = (dataset, model)
            if key not in groups:
                groups[key] = []
            groups[key].append((attack, os.path.join(baseline_dir, f)))
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    # Plotting
    for (dataset, model), items in groups.items():
        print(f"Plotting ROC for {dataset} - {model}...")
        plt.figure(figsize=(10, 8))
        
        # Sort items: 'ours' first, then alphabetically
        def sort_key(item):
            attack_name = item[0]
            if attack_name.lower() == 'ours':
                return (0, attack_name)  # ours gets priority 0
            else:
                return (1, attack_name)  # others get priority 1
        
        items.sort(key=sort_key)

        for attack, filepath in items:
            try:
                data = np.load(filepath)
                y_true = None
                y_score = None
                
                # Case 1: Standard RAPID format (acc, ROC_label, ROC_confidence_score)
                if 'ROC_label' in data and 'ROC_confidence_score' in data:
                    y_true = data['ROC_label']
                    y_score = data['ROC_confidence_score']
                
                # Case 2: Watson format (train_vals, heldout_vals, auc)
                elif 'train_vals' in data and 'heldout_vals' in data:
                    train_scores = data['train_vals']
                    heldout_scores = data['heldout_vals']
                    
                    # Ensure 1D
                    train_scores = np.asarray(train_scores).flatten()
                    heldout_scores = np.asarray(heldout_scores).flatten()
                    
                    # Create labels: 1 for training data (members), 0 for heldout data (non-members)
                    y_true = np.concatenate([np.ones_like(train_scores), np.zeros_like(heldout_scores)])
                    
                    # Watson/Loss-based attacks: Lower score (loss) means higher probability of membership.
                    # We negate the scores so that higher values correspond to members, which roc_curve expects.
                    y_score = -np.concatenate([train_scores, heldout_scores])
                
                # Case 3: Direct format (scores, labels)
                elif 'scores' in data and 'labels' in data:
                    y_score = data['scores']
                    y_true = data['labels']
                    
                    # Ensure 1D
                    y_score = np.asarray(y_score).flatten()
                    y_true = np.asarray(y_true).flatten()
                
                # Case 4: Pre-computed ROC curve (fpr, tpr, auc)
                elif 'fpr' in data and 'tpr' in data:
                    fpr = np.asarray(data['fpr'])
                    tpr = np.asarray(data['tpr'])
                    roc_auc = float(data['auc']) if 'auc' in data else auc(fpr, tpr)
                    
                    # Directly plot pre-computed ROC curve
                    plt.plot(fpr, tpr, lw=2, label=f'{attack}')
                    continue  # Skip the processing below
                
                else:
                    print(f"  Warning: File {filepath} does not contain recognized keys. Keys found: {list(data.files)}")
                    continue

                # Process if we have valid data (for Case 1-3)
                if y_true is not None and y_score is not None:
                    # Ensure y_score is 1D array
                    y_score = np.nan_to_num(y_score)
                    if y_score.ndim > 1:
                        y_score = y_score.flatten()
                    if y_true.ndim > 1:
                        y_true = y_true.flatten()

                    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
                    roc_auc = auc(fpr, tpr)
                    
                    # Plot
                    plt.plot(fpr, tpr, lw=2, label=f'{attack}')
                    
            except Exception as e:
                print(f"  Error reading {filepath}: {e}")
                
        plt.plot([1e-4, 1], [1e-4, 1], color='navy', lw=2, linestyle='--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([1e-4, 1.0])
        plt.ylim([1e-4, 1.0])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.legend(loc="lower right", fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_filename_png = f"{dataset}_{model}_roc_comparison.png"
        out_filename_pdf = f"{dataset}_{model}_roc_comparison.pdf"
        out_path_png = os.path.join(output_dir, out_filename_png)
        out_path_pdf = os.path.join(output_dir, out_filename_pdf)
        
        plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to {out_path_png}")
        
        plt.savefig(out_path_pdf, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to {out_path_pdf}")
        
        plt.close()

if __name__ == "__main__":
    # Assuming the script runs from RAPID-main directory
    plot_all_rocs(baseline_dir="baseline", output_dir="results/baseline_plots")
