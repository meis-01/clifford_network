#!/bin/bash
# Quick Reference Commands for Weight Initialization Comparison Study
# Usage: Source this file or copy individual commands

# PROJECT HOME
PROJECT_HOME="/home/mad07/clifford_network"

# ============================================================================
# BASIC COMMANDS
# ============================================================================

# Activate environment (if using venv/conda)
# source /home/mad07/pytorch_env/bin/activate

# Navigate to project
cd "$PROJECT_HOME"

# ============================================================================
# 1. TRAINING - RUN COMPARISON
# ============================================================================

# Train ALL initialization methods (RECOMMENDED FIRST RUN)
python -m src.autoencoder.comparison_train \
    --config configs/large_autoencoder.yaml \
    --log-level INFO

# Train ONLY specific methods (faster for testing)
python -m src.autoencoder.comparison_train \
    --config configs/large_autoencoder.yaml \
    --methods xavier he unitary \
    --log-level INFO

# Train with DEBUG logging (verbose)
python -m src.autoencoder.comparison_train \
    --config configs/large_autoencoder.yaml \
    --log-level DEBUG

# ============================================================================
# 2. MONITORING - CHECK PROGRESS
# ============================================================================

# Watch main comparison log in real-time
tail -f runs/weight_init_comparison/comparison.log

# Watch specific method's log
tail -f runs/weight_init_comparison/init_xavier/train.log

# Check what methods have completed
ls -la runs/weight_init_comparison/ | grep init_

# Show file sizes (training data)
du -sh runs/weight_init_comparison/init_*/

# ============================================================================
# 3. ANALYSIS - AFTER TRAINING COMPLETES
# ============================================================================

# Run full analysis with all methods
python -m src.autoencoder.comparison_analysis \
    --results-dir runs/weight_init_comparison \
    --methods random xavier he unitary trabelsi structured_preserve

# Run analysis with subset of methods
python -m src.autoencoder.comparison_analysis \
    --results-dir runs/weight_init_comparison \
    --methods xavier he unitary

# Save plots to custom directory
python -m src.autoencoder.comparison_analysis \
    --results-dir runs/weight_init_comparison \
    --methods random xavier he unitary trabelsi structured_preserve \
    --output-dir /home/mad07/clifford_network/runs/weight_init_comparison/analysis_results

# ============================================================================
# 4. VIEWING RESULTS
# ============================================================================

# View main comparison summary
cat runs/weight_init_comparison/comparison_summary.json | python -m json.tool

# View specific method summary
cat runs/weight_init_comparison/init_xavier/summary.json | python -m json.tool

# View training history
cat runs/weight_init_comparison/init_xavier/history.json | python -m json.tool | head -50

# List all generated plots
ls -lh runs/weight_init_comparison/plots/

# View comparison summary table (CSV)
column -t -s',' runs/weight_init_comparison/plots/comparison_summary.csv

# ============================================================================
# 5. ANALYSIS - EXTRACT METRICS
# ============================================================================

# Get best validation loss for each method
for method in random xavier he unitary trabelsi structured_preserve; do
    loss=$(cat runs/weight_init_comparison/init_$method/summary.json 2>/dev/null | grep best_val_loss | head -1 | grep -oP '[\d.]+')
    echo "$method: $loss"
done

# Show convergence speed (best epoch)
echo "Method              Best Epoch  Training Time (min)"
echo "---------------------------------------------------"
for method in random xavier he unitary trabelsi structured_preserve; do
    file="runs/weight_init_comparison/init_$method/summary.json"
    if [ -f "$file" ]; then
        epoch=$(cat "$file" | grep -oP '"best_epoch":\s*\K\d+')
        time=$(cat "$file" | grep -oP '"training_time_seconds":\s*\K[\d.]+' | xargs python -c "import sys; print(f'{float(sys.argv[1])/60:.2f}')" 2>/dev/null || echo "N/A")
        printf "%-20s %-11s %s\n" "$method" "$epoch" "$time"
    fi
done

# ============================================================================
# 6. TROUBLESHOOTING & UTILITIES
# ============================================================================

# Check if any training failed
cat runs/weight_init_comparison/comparison.log | grep -i "error\|failed"

# Show all methods that trained successfully
for method in random xavier he unitary trabelsi structured_preserve; do
    if [ -f "runs/weight_init_comparison/init_$method/summary.json" ]; then
        echo "✓ $method: OK"
    else
        echo "✗ $method: MISSING"
    fi
done

# Clean up old runs (CAREFUL - removes directories!)
# rm -rf runs/weight_init_comparison/

# Show last 20 lines of main log
tail -20 runs/weight_init_comparison/comparison.log

# Show training statistics (requires jq: sudo apt install jq)
for method in random xavier he unitary trabelsi structured_preserve; do
    file="runs/weight_init_comparison/init_$method/history.json"
    if [ -f "$file" ]; then
        echo "=== $method ==="
        head -3 "$file" && echo "..." && tail -3 "$file"
    fi
done

# ============================================================================
# 7. CONFIGURATION TWEAKS
# ============================================================================

# Edit large autoencoder config
vim configs/large_autoencoder.yaml

# Quick config changes:
# - Increase epochs: data.epochs: 150
# - Decrease batch size: data.batch_size: 2
# - Increase learning rate: training.learning_rate: 2.0e-4

# ============================================================================
# 8. ADVANCED WORKFLOWS
# ============================================================================

# Run comparison AND analysis (sequential)
python -m src.autoencoder.comparison_train --config configs/large_autoencoder.yaml && \
python -m src.autoencoder.comparison_analysis \
    --results-dir runs/weight_init_comparison \
    --methods random xavier he unitary trabelsi structured_preserve

# Archive results before cleaning
tar -czf runs/weight_init_comparison_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
    runs/weight_init_comparison/

# Create comparison report (requires converting to PDF separately)
python3 << 'EOF'
import json
from pathlib import Path

results_dir = Path("runs/weight_init_comparison")
summary = json.loads((results_dir / "comparison_summary.json").read_text())

print("WEIGHT INITIALIZATION COMPARISON REPORT")
print("=" * 60)
print(f"Timestamp: {summary['timestamp']}")
print(f"Methods tested: {len(summary['results'])}")
print("\nResults:")
for method, result in sorted(summary['results'].items()):
    if 'error' not in result:
        print(f"  {method:20s} Loss: {result['best_val_loss']:.6f} Epoch: {result['best_epoch']}")
EOF

# ============================================================================
# NOTES
# ============================================================================

# After training completes:
# 1. Review runs/weight_init_comparison/comparison_summary.json
# 2. Run analysis script to generate plots
# 3. Check comparison_overview.png for visual summary
# 4. Read comparison_summary.csv for tabular metrics
# 5. Look at detailed_loss_curves.png for training dynamics

# Typical workflow:
# 1. Start training in tmux/screen (it takes hours)
# 2. Monitor progress with: tail -f runs/weight_init_comparison/comparison.log
# 3. After training, run analysis
# 4. View plots and results in runs/weight_init_comparison/plots/
