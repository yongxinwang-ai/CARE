#!/usr/bin/env python3
"""
Validate PDB implementation structure without running
"""

import os
import ast
import sys

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} NOT FOUND")
        return False

def check_class_in_file(filepath, classname):
    """Check if a class exists in a Python file"""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == classname:
                print(f"  ✓ Found class: {classname}")
                return True
        print(f"  ✗ Class not found: {classname}")
        return False
    except Exception as e:
        print(f"  ✗ Error parsing file: {e}")
        return False

def check_function_in_file(filepath, funcname):
    """Check if a function exists in a Python file"""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == funcname:
                print(f"  ✓ Found function: {funcname}")
                return True
        print(f"  ✗ Function not found: {funcname}")
        return False
    except Exception as e:
        print(f"  ✗ Error parsing file: {e}")
        return False

def validate_pdb_implementation():
    """Validate the PDB implementation structure"""
    
    print("=" * 60)
    print("PDB Implementation Structure Validation")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Check core PDB modules
    print("\n1. Core PDB Modules:")
    print("-" * 40)
    
    modules = [
        ("verl/utils/pdb_actions.py", "Action definitions"),
        ("verl/utils/pdb_costs.py", "Cost models"),
        ("verl/utils/pdb_gains.py", "Information gain estimators"),
        ("verl/utils/pdb_dual.py", "Dual variable management"),
    ]
    
    for filepath, description in modules:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    # Check key classes in each module
    print("\n2. Key Classes:")
    print("-" * 40)
    
    classes_to_check = [
        ("verl/utils/pdb_actions.py", ["ActionType", "ActionSpace", "Action"]),
        ("verl/utils/pdb_costs.py", ["CostModel", "BudgetTracker"]),
        ("verl/utils/pdb_gains.py", ["GainEstimator", "EntropyGainEstimator", "CombinedGainEstimator"]),
        ("verl/utils/pdb_dual.py", ["DualVariableManager", "OptimalStopping", "PDBController"]),
    ]
    
    for filepath, classnames in classes_to_check:
        if os.path.exists(filepath):
            print(f"\nChecking {filepath}:")
            for classname in classnames:
                if not check_class_in_file(filepath, classname):
                    all_checks_passed = False
    
    # Check trainer integration
    print("\n3. Trainer Integration:")
    print("-" * 40)
    
    trainer_file = "verl/trainer/ray_trainer.py"
    if check_file_exists(trainer_file, "Ray trainer"):
        if not check_function_in_file(trainer_file, "_apply_pdb_variant"):
            all_checks_passed = False
    
    # Check configuration
    print("\n4. Configuration:")
    print("-" * 40)
    
    config_files = [
        ("examples/config.yaml", "Main config"),
        ("verl/trainer/config.py", "Config classes"),
    ]
    
    for filepath, description in config_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    # Check for PDB config in yaml
    if os.path.exists("examples/config.yaml"):
        with open("examples/config.yaml", 'r') as f:
            content = f.read()
            if "pdb_config:" in content:
                print("  ✓ Found pdb_config in config.yaml")
            else:
                print("  ✗ pdb_config not found in config.yaml")
                all_checks_passed = False
    
    # Check experiment scripts
    print("\n5. Experiment Scripts:")
    print("-" * 40)
    
    scripts = [
        "examples/qwen2_5_vl_7b_geo3k_pdb_grpo.sh",
        "examples/qwen2_5_vl_7b_geo3k_pdb_strict_grpo.sh",
        "examples/qwen2_5_vl_7b_geo3k_pdb_relaxed_grpo.sh",
    ]
    
    for script in scripts:
        if not check_file_exists(script, "Experiment script"):
            all_checks_passed = False
    
    # Check task documentation
    print("\n6. Documentation:")
    print("-" * 40)
    
    docs = [
        ("tasks/001-pdb.md", "Task documentation"),
        ("ROADMAP.md", "Roadmap"),
    ]
    
    for filepath, description in docs:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✓ All structure checks PASSED!")
        print("The PDB implementation structure is complete.")
    else:
        print("✗ Some checks FAILED.")
        print("Please review the missing components.")
    print("=" * 60)
    
    return all_checks_passed

def check_imports():
    """Check if modules can be imported (will fail without dependencies)"""
    print("\n7. Import Test (Expected to fail without torch):")
    print("-" * 40)
    
    modules_to_import = [
        "verl.utils.pdb_actions",
        "verl.utils.pdb_costs",
        "verl.utils.pdb_gains",
        "verl.utils.pdb_dual",
    ]
    
    for module in modules_to_import:
        try:
            exec(f"import {module}")
            print(f"  ✓ Successfully imported: {module}")
        except ImportError as e:
            print(f"  ✗ Cannot import {module}: {e}")
            print(f"    (This is expected without torch/dependencies)")

if __name__ == "__main__":
    success = validate_pdb_implementation()
    check_imports()
    sys.exit(0 if success else 1)