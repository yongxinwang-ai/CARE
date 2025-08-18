#!/usr/bin/env python3
"""
Basic tests for PDB (Primal-Dual Budgeter) components that don't require torch
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPDBBasicImports(unittest.TestCase):
    """Test basic imports and structure"""
    
    def test_action_module_exists(self):
        """Test that action module can be imported"""
        try:
            from verl.utils import pdb_actions
            self.assertTrue(hasattr(pdb_actions, 'ActionType'))
            self.assertTrue(hasattr(pdb_actions, 'ActionSpace'))
            self.assertTrue(hasattr(pdb_actions, 'AnswerAction'))
        except ImportError as e:
            self.fail(f"Failed to import pdb_actions: {e}")
    
    def test_cost_module_exists(self):
        """Test that cost module can be imported"""
        try:
            from verl.utils import pdb_costs
            self.assertTrue(hasattr(pdb_costs, 'CostModel'))
            self.assertTrue(hasattr(pdb_costs, 'BudgetTracker'))
        except ImportError as e:
            self.fail(f"Failed to import pdb_costs: {e}")
    
    def test_gain_module_exists(self):
        """Test that gain module can be imported"""
        try:
            from verl.utils import pdb_gains
            self.assertTrue(hasattr(pdb_gains, 'GainEstimator'))
            self.assertTrue(hasattr(pdb_gains, 'EntropyGainEstimator'))
        except ImportError as e:
            self.fail(f"Failed to import pdb_gains: {e}")
    
    def test_dual_module_exists(self):
        """Test that dual module can be imported"""
        try:
            from verl.utils import pdb_dual
            self.assertTrue(hasattr(pdb_dual, 'DualVariableManager'))
            self.assertTrue(hasattr(pdb_dual, 'PDBController'))
        except ImportError as e:
            self.fail(f"Failed to import pdb_dual: {e}")


class TestPDBConfig(unittest.TestCase):
    """Test configuration structure"""
    
    def test_algorithm_config_has_pdb(self):
        """Test that AlgorithmConfig includes PDB config"""
        from verl.trainer.config import AlgorithmConfig
        
        config = AlgorithmConfig()
        self.assertTrue(hasattr(config, 'pdb_config'))
        self.assertIsInstance(config.pdb_config, dict)
        
        # Check key fields exist
        self.assertIn('visual_budget', config.pdb_config)
        self.assertIn('text_budget', config.pdb_config)
        self.assertIn('lambda_v_init', config.pdb_config)
        self.assertIn('lambda_t_init', config.pdb_config)
    
    def test_grpo_variant_includes_pdb(self):
        """Test that grpo_variant can be set to pdb"""
        from verl.trainer.config import AlgorithmConfig
        
        config = AlgorithmConfig()
        self.assertTrue(hasattr(config, 'grpo_variant'))
        
        # Should be able to set to pdb
        config.grpo_variant = 'pdb'
        self.assertEqual(config.grpo_variant, 'pdb')


class TestPDBActionStructure(unittest.TestCase):
    """Test action structure without torch"""
    
    def test_action_types(self):
        """Test ActionType enum"""
        from verl.utils.pdb_actions import ActionType
        
        # Check that all action types exist
        self.assertTrue(hasattr(ActionType, 'ANS'))
        self.assertTrue(hasattr(ActionType, 'TEXT'))
        self.assertTrue(hasattr(ActionType, 'CROP'))
        self.assertTrue(hasattr(ActionType, 'DRAW'))
        self.assertTrue(hasattr(ActionType, 'TOOL'))
        self.assertTrue(hasattr(ActionType, 'STOP'))
    
    def test_draw_primitives(self):
        """Test DrawPrimitive enum"""
        from verl.utils.pdb_actions import DrawPrimitive
        
        # Check that draw primitives exist
        self.assertTrue(hasattr(DrawPrimitive, 'LINE'))
        self.assertTrue(hasattr(DrawPrimitive, 'ANGLE'))
        self.assertTrue(hasattr(DrawPrimitive, 'MEASURE'))
    
    def test_action_creation(self):
        """Test creating actions without torch"""
        from verl.utils.pdb_actions import (
            AnswerAction, TextAction, CropAction, StopAction
        )
        
        # Create actions
        ans = AnswerAction()
        self.assertIsNotNone(ans)
        
        text = TextAction(num_tokens=10)
        self.assertEqual(text.num_tokens, 10)
        
        crop = CropAction((0.1, 0.2, 0.3, 0.4))
        self.assertEqual(crop.bbox, (0.1, 0.2, 0.3, 0.4))
        
        stop = StopAction()
        self.assertIsNotNone(stop)


class TestPDBCostStructure(unittest.TestCase):
    """Test cost model structure without torch"""
    
    def test_cost_model_creation(self):
        """Test creating cost model"""
        from verl.utils.pdb_costs import CostModel
        
        config = {
            "cost_per_patch": 1,
            "cost_crop_fixed": 32,
            "cost_draw_per_primitive": 4,
            "cost_per_text_token": 1
        }
        
        model = CostModel(config)
        self.assertIsNotNone(model)
        self.assertEqual(model.cost_per_patch, 1)
        self.assertEqual(model.cost_crop_fixed, 32)
    
    def test_budget_tracker_creation(self):
        """Test creating budget tracker"""
        from verl.utils.pdb_costs import BudgetTracker
        
        tracker = BudgetTracker(visual_budget=100, text_budget=50)
        self.assertEqual(tracker.visual_budget, 100)
        self.assertEqual(tracker.text_budget, 50)
        self.assertEqual(tracker.visual_used, 0)
        self.assertEqual(tracker.text_used, 0)
        
        # Test can_afford
        self.assertTrue(tracker.can_afford(50, 25))
        self.assertFalse(tracker.can_afford(200, 25))


class TestPDBIntegration(unittest.TestCase):
    """Test integration points"""
    
    def test_trainer_has_pdb_method(self):
        """Test that trainer has PDB variant method"""
        import inspect
        from verl.trainer.ray_trainer import RayPPOTrainer
        
        # Check that _apply_pdb_variant method exists
        self.assertTrue(hasattr(RayPPOTrainer, '_apply_pdb_variant'))
        
        # Check method signature
        method = getattr(RayPPOTrainer, '_apply_pdb_variant')
        self.assertTrue(callable(method))
    
    def test_experiment_scripts_exist(self):
        """Test that experiment scripts were created"""
        import os
        
        script_dir = "examples"
        expected_scripts = [
            "qwen2_5_vl_7b_geo3k_pdb_grpo.sh",
            "qwen2_5_vl_7b_geo3k_pdb_strict_grpo.sh",
            "qwen2_5_vl_7b_geo3k_pdb_relaxed_grpo.sh"
        ]
        
        for script in expected_scripts:
            script_path = os.path.join(script_dir, script)
            self.assertTrue(
                os.path.exists(script_path),
                f"Script {script_path} does not exist"
            )


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPDBBasicImports))
    suite.addTests(loader.loadTestsFromTestCase(TestPDBConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestPDBActionStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestPDBCostStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestPDBIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)