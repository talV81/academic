#!/usr/bin/env python3
"""
Extract accuracy results from log files in the output directory.
"""
import os
import re
import json
from pathlib import Path
from collections import defaultdict, OrderedDict

def extract_accuracy_from_log(log_file):
    """Extract accuracy results from a log file."""
    results = {}
    
    if not os.path.exists(log_file):
        return results
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract test accuracy (final evaluation)
    test_match = re.search(r'=> result\s*\n.*?\* accuracy: ([\d.]+)%', content, re.DOTALL)
    if test_match:
        results['test_accuracy'] = float(test_match.group(1))
    
    # Extract validation accuracies during training (if available)
    val_matches = re.findall(r'val/accuracy[:\s]+([\d.]+)', content)
    if val_matches:
        results['val_accuracies'] = [float(x) for x in val_matches]
        results['best_val_accuracy'] = max(results['val_accuracies'])
        results['final_val_accuracy'] = results['val_accuracies'][-1] if results['val_accuracies'] else None
    
    # Extract test accuracies during training (if available)
    test_matches = re.findall(r'test/accuracy[:\s]+([\d.]+)', content)
    if test_matches:
        results['test_accuracies'] = [float(x) for x in test_matches]
    
    # Extract per-class accuracy if available
    perclass_match = re.search(r'\* average: ([\d.]+)%', content)
    if perclass_match:
        results['perclass_accuracy'] = float(perclass_match.group(1))
    
    # Extract macro F1 if available
    f1_match = re.search(r'\* macro_f1: ([\d.]+)%', content)
    if f1_match:
        results['macro_f1'] = float(f1_match.group(1))
    
    # Extract error rate if available
    err_match = re.search(r'\* error: ([\d.]+)%', content)
    if err_match:
        results['error_rate'] = float(err_match.group(1))
    
    return results

def collect_all_results(output_dir):
    """Collect all accuracy results from log files in the output directory."""
    output_path = Path(output_dir)
    all_results = OrderedDict()
    
    # Find all log.txt files
    for log_file in output_path.rglob('log.txt'):
        # Get relative path from output_dir
        rel_path = log_file.relative_to(output_path)
        results = extract_accuracy_from_log(log_file)
        
        if results:
            all_results[str(rel_path)] = results
    
    return all_results

def print_results_summary(all_results):
    """Print a summary of all results."""
    print("=" * 80)
    print("ACCURACY RESULTS SUMMARY")
    print("=" * 80)
    
    for log_path, results in all_results.items():
        print(f"\n📁 {log_path}")
        print("-" * 80)
        
        if 'test_accuracy' in results:
            print(f"  Test Accuracy: {results['test_accuracy']:.2f}%")
        
        if 'best_val_accuracy' in results:
            print(f"  Best Val Accuracy: {results['best_val_accuracy']:.2f}%")
        
        if 'final_val_accuracy' in results:
            print(f"  Final Val Accuracy: {results['final_val_accuracy']:.2f}%")
        
        if 'perclass_accuracy' in results:
            print(f"  Per-Class Accuracy: {results['perclass_accuracy']:.2f}%")
        
        if 'macro_f1' in results:
            print(f"  Macro F1: {results['macro_f1']:.2f}%")
        
        if 'error_rate' in results:
            print(f"  Error Rate: {results['error_rate']:.2f}%")
        
        if 'val_accuracies' in results:
            print(f"  Val Accuracies (epochs): {len(results['val_accuracies'])} evaluations")
            if len(results['val_accuracies']) <= 10:
                print(f"    {results['val_accuracies']}")
            else:
                print(f"    First 5: {results['val_accuracies'][:5]}")
                print(f"    Last 5: {results['val_accuracies'][-5:]}")

def save_results_json(all_results, output_file='accuracy_results.json'):
    """Save results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract accuracy results from log files')
    parser.add_argument('output_dir', type=str, help='Output directory containing log files')
    parser.add_argument('--json', type=str, help='Save results to JSON file')
    parser.add_argument('--csv', type=str, help='Save results to CSV file')
    
    args = parser.parse_args()
    
    all_results = collect_all_results(args.output_dir)
    
    if not all_results:
        print(f"No results found in {args.output_dir}")
        return
    
    print_results_summary(all_results)
    
    if args.json:
        save_results_json(all_results, args.json)
    
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Log Path', 'Test Accuracy', 'Best Val Accuracy', 'Final Val Accuracy', 
                           'Per-Class Accuracy', 'Macro F1', 'Error Rate'])
            for log_path, results in all_results.items():
                writer.writerow([
                    log_path,
                    results.get('test_accuracy', ''),
                    results.get('best_val_accuracy', ''),
                    results.get('final_val_accuracy', ''),
                    results.get('perclass_accuracy', ''),
                    results.get('macro_f1', ''),
                    results.get('error_rate', '')
                ])
        print(f"\n✅ Results saved to {args.csv}")

if __name__ == '__main__':
    main()


