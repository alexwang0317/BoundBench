import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from pathlib import Path
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data(filepath):
    """Load JSON data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_labels(data, input_filename=None, label_type='binary'):
    """Analyze and create visualizations for judge labels and differences."""
    
    # Extract relevant fields
    judge_labels = [entry['judge_label'] for entry in data]
    true_label_field = f'true_label_{label_type}'
    true_labels = [entry[true_label_field] for entry in data]
    concepts = [entry['Concept'] for entry in data]
    question_nums = [entry['question_num'] for entry in data]
    
    # Calculate differences (true_label - judge_label)
    differences = [true - judge for true, judge in zip(true_labels, judge_labels)]
    
    # Create output directory name based on input file and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if input_filename:
        base_name = Path(input_filename).stem  # Get filename without extension
        dir_name = f"{base_name}_{label_type}_{timestamp}"
    else:
        dir_name = f"analysis_{label_type}_{timestamp}"
    
    # Create output directory for plots in the same location as input file
    if input_filename:
        input_path = Path(input_filename)
        output_dir = input_path.parent / dir_name
    else:
        output_dir = Path(dir_name)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Comparing against: {true_label_field}")
    print(f"Total entries: {len(data)}")
    print(f"Judge labels range: {min(judge_labels)} to {max(judge_labels)}")
    print(f"True labels range: {min(true_labels)} to {max(true_labels)}")
    print(f"Differences range: {min(differences)} to {max(differences)}")
    
    # 1. Distribution of judge_label values
    plot_judge_label_distribution(judge_labels, output_dir)
    
    # 2. Overall difference distribution
    plot_overall_difference_distribution(differences, output_dir)
    
    # 3. Difference distribution by true_label
    plot_difference_by_true_label(true_labels, differences, output_dir)
    
    # 4. Difference distribution by concept
    plot_difference_by_concept(concepts, differences, output_dir)
    
    # 5. Difference distribution by question number
    plot_difference_by_question(question_nums, differences, output_dir)
    
    # 6. Confusion matrix style visualization
    plot_confusion_matrix(true_labels, judge_labels, output_dir)
    
    # 7. Summary statistics
    print_summary_statistics(true_labels, judge_labels, differences, concepts, question_nums)
    
    print(f"\nAll plots saved to: {output_dir}")

def plot_judge_label_distribution(judge_labels, output_dir):
    """Plot distribution of judge_label values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    label_counts = Counter(judge_labels)
    sorted_labels = sorted(label_counts.keys())
    counts = [label_counts[label] for label in sorted_labels]
    
    ax1.bar(sorted_labels, counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Judge Label', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Judge Labels', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (label, count) in enumerate(zip(sorted_labels, counts)):
        ax1.text(label, count, str(count), ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(counts, labels=sorted_labels, autopct='%1.1f%%', startangle=90, 
            colors=sns.color_palette('pastel'))
    ax2.set_title('Judge Label Proportions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'judge_label_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Judge label distribution plot saved")

def plot_overall_difference_distribution(differences, output_dir):
    """Plot overall distribution of differences (true_label - judge_label)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    diff_counts = Counter(differences)
    sorted_diffs = sorted(diff_counts.keys())
    counts = [diff_counts[diff] for diff in sorted_diffs]
    
    # Color bars based on whether difference is negative, zero, or positive
    colors = ['red' if d < 0 else 'green' if d > 0 else 'gray' for d in sorted_diffs]
    
    ax1.bar(sorted_diffs, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Difference (True Label - Judge Label)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Overall Distribution of Label Differences', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for diff, count in zip(sorted_diffs, counts):
        ax1.text(diff, count, str(count), ha='center', va='bottom')
    
    # Histogram with KDE
    ax2.hist(differences, bins=range(int(min(differences))-1, int(max(differences))+2), 
             alpha=0.7, color='steelblue', edgecolor='black', density=True)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Agreement')
    ax2.set_xlabel('Difference (True Label - Judge Label)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Difference Distribution (Normalized)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_difference_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Overall difference distribution plot saved")

def plot_difference_by_true_label(true_labels, differences, output_dir):
    """Plot difference distribution for each true_label value."""
    unique_true_labels = sorted(set(true_labels))
    n_labels = len(unique_true_labels)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_labels, figsize=(6*n_labels, 5))
    if n_labels == 1:
        axes = [axes]
    
    # Get global range for consistent x-axis
    all_diffs = set(differences)
    diff_range = range(int(min(differences)), int(max(differences))+1)
    
    for idx, true_label in enumerate(unique_true_labels):
        # Filter differences for this true_label
        label_diffs = [diff for true, diff in zip(true_labels, differences) if true == true_label]
        
        # Count occurrences
        diff_counts = Counter(label_diffs)
        sorted_diffs = sorted(diff_range)
        counts = [diff_counts.get(diff, 0) for diff in sorted_diffs]
        
        # Color bars
        colors = ['red' if d < 0 else 'green' if d > 0 else 'gray' for d in sorted_diffs]
        
        axes[idx].bar(sorted_diffs, counts, color=colors, alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel('Difference', fontsize=11)
        axes[idx].set_ylabel('Count', fontsize=11)
        axes[idx].set_title(f'True Label = {true_label}\n(n={len(label_diffs)})', 
                           fontsize=12, fontweight='bold')
        axes[idx].axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars (only if count > 0)
        for diff, count in zip(sorted_diffs, counts):
            if count > 0:
                axes[idx].text(diff, count, str(count), ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Difference Distribution by True Label', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'difference_by_true_label.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Difference by true label plot saved")

def plot_difference_by_concept(concepts, differences, output_dir):
    """Plot difference distribution for each concept."""
    unique_concepts = sorted(set(concepts))
    n_concepts = len(unique_concepts)
    
    print(f"\nFound {n_concepts} unique concepts")
    
    # Determine grid layout
    n_cols = min(3, n_concepts)
    n_rows = (n_concepts + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Flatten axes array for easier iteration
    if n_concepts == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_concepts > n_cols else axes
    
    # Get global range for consistent x-axis
    diff_range = range(int(min(differences)), int(max(differences))+1)
    
    for idx, concept in enumerate(unique_concepts):
        # Filter differences for this concept
        concept_diffs = [diff for conc, diff in zip(concepts, differences) if conc == concept]
        
        # Count occurrences
        diff_counts = Counter(concept_diffs)
        sorted_diffs = sorted(diff_range)
        counts = [diff_counts.get(diff, 0) for diff in sorted_diffs]
        
        # Color bars
        colors = ['red' if d < 0 else 'green' if d > 0 else 'gray' for d in sorted_diffs]
        
        ax = axes[idx] if n_concepts > 1 else axes[0]
        ax.bar(sorted_diffs, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Difference', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        
        # Truncate long concept names for title
        concept_title = concept if len(concept) <= 30 else concept[:27] + '...'
        ax.set_title(f'{concept_title}\n(n={len(concept_diffs)})', 
                     fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars (only if count > 0)
        for diff, count in zip(sorted_diffs, counts):
            if count > 0:
                ax.text(diff, count, str(count), ha='center', va='bottom', fontsize=8)
    
    # Hide any unused subplots
    for idx in range(n_concepts, len(axes) if hasattr(axes, '__len__') else 1):
        if hasattr(axes, '__len__'):
            axes[idx].set_visible(False)
    
    plt.suptitle('Difference Distribution by Concept', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(output_dir / 'difference_by_concept.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Difference by concept plot saved")

def plot_difference_by_question(question_nums, differences, output_dir):
    """Plot difference distribution for each question number."""
    unique_questions = sorted(set(question_nums))
    n_questions = len(unique_questions)
    
    print(f"\nFound {n_questions} unique questions")
    
    # Determine grid layout
    n_cols = min(5, n_questions)
    n_rows = (n_questions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    # Flatten axes array for easier iteration
    if n_questions == 1:
        axes = [axes]
    elif n_questions <= n_cols:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    # Get global range for consistent x-axis
    diff_range = range(int(min(differences)), int(max(differences))+1)
    
    for idx, question_num in enumerate(unique_questions):
        # Filter differences for this question
        question_diffs = [diff for qnum, diff in zip(question_nums, differences) if qnum == question_num]
        
        # Count occurrences
        diff_counts = Counter(question_diffs)
        sorted_diffs = sorted(diff_range)
        counts = [diff_counts.get(diff, 0) for diff in sorted_diffs]
        
        # Color bars
        colors = ['red' if d < 0 else 'green' if d > 0 else 'gray' for d in sorted_diffs]
        
        ax = axes[idx] if n_questions > 1 else axes[0]
        ax.bar(sorted_diffs, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Difference', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(f'Question {question_num}\n(n={len(question_diffs)})', 
                     fontsize=10, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars (only if count > 0)
        for diff, count in zip(sorted_diffs, counts):
            if count > 0:
                ax.text(diff, count, str(count), ha='center', va='bottom', fontsize=7)
    
    # Hide any unused subplots
    for idx in range(n_questions, len(axes) if hasattr(axes, '__len__') else 1):
        if hasattr(axes, '__len__'):
            axes[idx].set_visible(False)
    
    plt.suptitle('Difference Distribution by Question Number', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(output_dir / 'difference_by_question.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Difference by question number plot saved")

def plot_confusion_matrix(true_labels, judge_labels, output_dir):
    """Create a confusion matrix-style visualization."""
    from matplotlib.colors import LinearSegmentedColormap
    
    # Get unique labels
    all_labels = sorted(set(true_labels + judge_labels))
    n_labels = len(all_labels)
    
    # Create confusion matrix
    conf_matrix = np.zeros((n_labels, n_labels))
    for true, judge in zip(true_labels, judge_labels):
        true_idx = all_labels.index(true)
        judge_idx = all_labels.index(judge)
        conf_matrix[true_idx, judge_idx] += 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Custom colormap: white to blue
    cmap = LinearSegmentedColormap.from_list('custom', ['white', 'lightblue', 'darkblue'])
    
    # Plot heatmap
    im = ax.imshow(conf_matrix, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20, fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))
    ax.set_xticklabels(all_labels)
    ax.set_yticklabels(all_labels)
    
    ax.set_xlabel('Judge Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix: True vs Judge Labels', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(n_labels):
        for j in range(n_labels):
            count = int(conf_matrix[i, j])
            if count > 0:
                # Highlight diagonal (perfect matches) in bold
                weight = 'bold' if i == j else 'normal'
                color = 'white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black'
                ax.text(j, i, str(count), ha='center', va='center', 
                       fontweight=weight, fontsize=12, color=color)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrix plot saved")

def print_summary_statistics(true_labels, judge_labels, differences, concepts, question_nums):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Overall accuracy
    correct = sum(1 for t, j in zip(true_labels, judge_labels) if t == j)
    accuracy = correct / len(true_labels) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{len(true_labels)})")
    
    # Difference statistics
    print(f"\nDifference Statistics:")
    print(f"  Mean difference: {np.mean(differences):.3f}")
    print(f"  Std deviation: {np.std(differences):.3f}")
    print(f"  Median difference: {np.median(differences):.3f}")
    
    # Breakdown by difference value
    print(f"\nDifference Breakdown:")
    diff_counts = Counter(differences)
    for diff in sorted(diff_counts.keys()):
        count = diff_counts[diff]
        pct = count / len(differences) * 100
        interpretation = ""
        if diff < 0:
            interpretation = "(Judge over-predicted)"
        elif diff > 0:
            interpretation = "(Judge under-predicted)"
        else:
            interpretation = "(Perfect match)"
        print(f"  Diff {diff:+.1f}: {count:4d} ({pct:5.1f}%) {interpretation}")
    
    # Breakdown by true label
    print(f"\nAccuracy by True Label:")
    for true_label in sorted(set(true_labels)):
        label_correct = sum(1 for t, j in zip(true_labels, judge_labels) 
                          if t == true_label and t == j)
        label_total = sum(1 for t in true_labels if t == true_label)
        label_acc = label_correct / label_total * 100 if label_total > 0 else 0
        print(f"  True Label {true_label}: {label_acc:.2f}% ({label_correct}/{label_total})")
    
    # Breakdown by concept
    print(f"\nAccuracy by Concept:")
    unique_concepts = sorted(set(concepts))
    for concept in unique_concepts:
        concept_indices = [i for i, c in enumerate(concepts) if c == concept]
        concept_correct = sum(1 for i in concept_indices 
                            if true_labels[i] == judge_labels[i])
        concept_total = len(concept_indices)
        concept_acc = concept_correct / concept_total * 100 if concept_total > 0 else 0
        print(f"  {concept}: {concept_acc:.2f}% ({concept_correct}/{concept_total})")
    
    # Breakdown by question number
    print(f"\nAccuracy by Question Number:")
    unique_questions = sorted(set(question_nums))
    for question_num in unique_questions:
        question_indices = [i for i, q in enumerate(question_nums) if q == question_num]
        question_correct = sum(1 for i in question_indices 
                              if true_labels[i] == judge_labels[i])
        question_total = len(question_indices)
        question_acc = question_correct / question_total * 100 if question_total > 0 else 0
        print(f"  Question {question_num}: {question_acc:.2f}% ({question_correct}/{question_total})")
    
    print("\n" + "="*70)

def main():
    """Main function to run analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze judge evaluation data and generate visualizations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.json
  %(prog)s offline_judging/judge_evals/responses/questions_10_tri.json
  %(prog)s data.json --output-dir ./my_plots
  %(prog)s data.json --label-type hexanary
  %(prog)s data.json -l trinary -v
        """
    )
    
    parser.add_argument(
        'json_file',
        type=str,
        help='Path to the JSON file containing evaluation data'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: offline_judging/judge_evals/analysis_plots)'
    )
    
    parser.add_argument(
        '-l', '--label-type',
        type=str,
        choices=['binary', 'trinary', 'hexanary'],
        default='binary',
        help='Type of true label to compare against judge label (default: binary)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose output'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    filepath = Path(args.json_file)
    if not filepath.exists():
        parser.error(f"File not found: {args.json_file}")
    
    if args.verbose:
        print(f"\nLoading data from: {filepath}")
        print(f"Using label type: {args.label_type}")
    
    data = load_data(filepath)
    
    if args.verbose:
        print(f"Loaded {len(data)} entries")
        print("\nStarting analysis...")
    
    # Run analysis with custom output directory if specified
    if args.output_dir:
        analyze_labels_with_custom_output(data, args.output_dir, label_type=args.label_type)
    else:
        analyze_labels(data, input_filename=args.json_file, label_type=args.label_type)
    
    print("\n✓ Analysis complete!")

def analyze_labels_with_custom_output(data, output_dir_path, label_type='binary'):
    """Analyze and create visualizations with custom output directory."""
    
    # Extract relevant fields
    judge_labels = [entry['judge_label'] for entry in data]
    true_label_field = f'true_label_{label_type}'
    true_labels = [entry[true_label_field] for entry in data]
    concepts = [entry['Concept'] for entry in data]
    question_nums = [entry['question_num'] for entry in data]
    
    # Calculate differences (true_label - judge_label)
    differences = [true - judge for true, judge in zip(true_labels, judge_labels)]
    
    # Create output directory for plots
    output_dir = Path(output_dir_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Comparing against: {true_label_field}")
    print(f"Total entries: {len(data)}")
    print(f"Judge labels range: {min(judge_labels)} to {max(judge_labels)}")
    print(f"True labels range: {min(true_labels)} to {max(true_labels)}")
    print(f"Differences range: {min(differences)} to {max(differences)}")
    
    # 1. Distribution of judge_label values
    plot_judge_label_distribution(judge_labels, output_dir)
    
    # 2. Overall difference distribution
    plot_overall_difference_distribution(differences, output_dir)
    
    # 3. Difference distribution by true_label
    plot_difference_by_true_label(true_labels, differences, output_dir)
    
    # 4. Difference distribution by concept
    plot_difference_by_concept(concepts, differences, output_dir)
    
    # 5. Difference distribution by question number
    plot_difference_by_question(question_nums, differences, output_dir)
    
    # 6. Confusion matrix style visualization
    plot_confusion_matrix(true_labels, judge_labels, output_dir)
    
    # 7. Summary statistics
    print_summary_statistics(true_labels, judge_labels, differences, concepts, question_nums)
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()

