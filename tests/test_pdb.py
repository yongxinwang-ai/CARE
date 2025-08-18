#!/usr/bin/env python3
"""
Unit tests for PDB (Primal-Dual Budgeter) components
"""

import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl.utils.pdb_actions import (
    ActionType, ActionSpace, AnswerAction, TextAction, 
    CropAction, DrawAction, StopAction, DrawPrimitive
)
from verl.utils.pdb_costs import CostModel, BudgetTracker
from verl.utils.pdb_gains import (
    EntropyGainEstimator, ConsistencyGainEstimator, 
    TaskGainEstimator, CombinedGainEstimator
)
from verl.utils.pdb_dual import DualVariableManager, OptimalStopping, PDBController


class TestPDBActions(unittest.TestCase):
    """Test action space and action definitions"""
    
    def setUp(self):
        self.config = {
            "enable_crop": True,
            "enable_draw": True,
            "enable_tool": False,
            "max_crop_candidates": 5,
            "max_draw_primitives": 3
        }
        self.action_space = ActionSpace(self.config)
    
    def test_action_creation(self):
        """Test creating different action types"""
        ans_action = AnswerAction()
        self.assertEqual(ans_action.action_type, ActionType.ANS)
        
        text_action = TextAction(num_tokens=10)
        self.assertEqual(text_action.action_type, ActionType.TEXT)
        self.assertEqual(text_action.num_tokens, 10)
        
        crop_action = CropAction((0.1, 0.1, 0.5, 0.5))
        self.assertEqual(crop_action.action_type, ActionType.CROP)
        self.assertEqual(crop_action.bbox, (0.1, 0.1, 0.5, 0.5))
        
        stop_action = StopAction()
        self.assertEqual(stop_action.action_type, ActionType.STOP)
    
    def test_get_available_actions(self):
        """Test generating available actions"""
        state = {"task_type": "geometry"}
        image_features = torch.randn(1, 256, 768)
        
        actions = self.action_space.get_available_actions(state, image_features)
        
        # Check that we have various action types
        action_types = [a.action_type for a in actions]
        self.assertIn(ActionType.ANS, action_types)
        self.assertIn(ActionType.STOP, action_types)
        self.assertIn(ActionType.TEXT, action_types)
        self.assertIn(ActionType.CROP, action_types)
        self.assertIn(ActionType.DRAW, action_types)
        
        # Tool should not be included since it's disabled
        self.assertNotIn(ActionType.TOOL, action_types)
    
    def test_crop_candidates_generation(self):
        """Test crop candidate generation"""
        state = {}
        image_features = torch.randn(1, 256, 768)
        
        candidates = self.action_space._generate_crop_candidates(state, image_features)
        
        # Should have grid-based crops
        self.assertGreater(len(candidates), 0)
        
        # All should be CropAction
        for c in candidates:
            self.assertIsInstance(c, CropAction)
            # Check bbox format
            self.assertEqual(len(c.bbox), 4)
            # Check bbox values are in [0, 1]
            for val in c.bbox:
                self.assertGreaterEqual(val, 0)
                self.assertLessEqual(val, 1)


class TestPDBCosts(unittest.TestCase):
    """Test cost models"""
    
    def setUp(self):
        self.config = {
            "cost_per_patch": 1,
            "cost_crop_fixed": 32,
            "cost_draw_per_primitive": 4,
            "cost_per_text_token": 1,
            "cost_tool_ocr": 64,
            "cost_tool_detect": 48
        }
        self.cost_model = CostModel(self.config)
    
    def test_answer_cost(self):
        """Test cost computation for answer action"""
        action = AnswerAction()
        visual_cost, text_cost = self.cost_model.compute_cost(action, {"answer_type": "short"})
        
        self.assertEqual(visual_cost, 0.0)
        self.assertGreater(text_cost, 0)  # Should have text cost
    
    def test_text_cost(self):
        """Test cost computation for text action"""
        action = TextAction(num_tokens=20)
        visual_cost, text_cost = self.cost_model.compute_cost(action)
        
        self.assertEqual(visual_cost, 0.0)
        self.assertEqual(text_cost, 20 * self.config["cost_per_text_token"])
    
    def test_crop_cost(self):
        """Test cost computation for crop action"""
        action = CropAction((0.0, 0.0, 0.5, 0.5))  # Half image
        visual_cost, text_cost = self.cost_model.compute_cost(action)
        
        self.assertGreater(visual_cost, self.config["cost_crop_fixed"])
        self.assertEqual(text_cost, 0.0)
    
    def test_draw_cost(self):
        """Test cost computation for draw action"""
        action = DrawAction(DrawPrimitive.LINE, {"points": [(0, 0), (1, 1)]})
        visual_cost, text_cost = self.cost_model.compute_cost(action)
        
        self.assertGreaterEqual(visual_cost, self.config["cost_draw_per_primitive"])
        self.assertEqual(text_cost, 0.0)
    
    def test_budget_tracker(self):
        """Test budget tracking"""
        tracker = BudgetTracker(visual_budget=100, text_budget=50)
        
        # Check initial state
        self.assertTrue(tracker.can_afford(50, 20))
        self.assertFalse(tracker.can_afford(150, 20))
        self.assertFalse(tracker.can_afford(50, 60))
        
        # Consume some budget
        action = TextAction(10)
        tracker.consume(action, 0, 10)
        
        self.assertEqual(tracker.visual_used, 0)
        self.assertEqual(tracker.text_used, 10)
        
        # Check remaining
        remaining_v, remaining_t = tracker.get_remaining()
        self.assertEqual(remaining_v, 100)
        self.assertEqual(remaining_t, 40)


class TestPDBGains(unittest.TestCase):
    """Test information gain estimators"""
    
    def setUp(self):
        self.config = {
            "gain_temperature": 1.0,
            "gain_estimators": ["entropy", "consistency", "task"],
            "gain_weights": [0.4, 0.3, 0.3]
        }
    
    def test_entropy_gain_estimator(self):
        """Test entropy-based gain estimation"""
        estimator = EntropyGainEstimator(self.config)
        
        state = {"answer_entropy": 1.0}
        
        # Direct answer should reduce entropy significantly
        ans_action = AnswerAction()
        gain = estimator.estimate_gain(ans_action, state)
        self.assertGreater(gain, 0.5)
        
        # Text generation should reduce entropy moderately
        text_action = TextAction(10)
        gain = estimator.estimate_gain(text_action, state)
        self.assertGreater(gain, 0)
        self.assertLess(gain, 1.0)
    
    def test_consistency_gain_estimator(self):
        """Test consistency-based gain estimation"""
        estimator = ConsistencyGainEstimator(self.config)
        
        state = {"answer_consistency": 0.5}
        
        # Text should improve consistency
        text_action = TextAction(10)
        gain = estimator.estimate_gain(text_action, state)
        self.assertGreater(gain, 0)
        
        # Crop should improve consistency
        crop_action = CropAction((0.1, 0.1, 0.5, 0.5))
        gain = estimator.estimate_gain(crop_action, state)
        self.assertGreater(gain, 0)
    
    def test_task_gain_estimator(self):
        """Test task-specific gain estimation"""
        estimator = TaskGainEstimator(self.config)
        
        # Geometry task
        state = {"task_type": "geometry"}
        draw_action = DrawAction(DrawPrimitive.ANGLE, {})
        gain = estimator.estimate_gain(draw_action, state)
        self.assertGreater(gain, 0.3)  # Drawing should be valuable for geometry
        
        # Text reading task
        state = {"task_type": "text_reading"}
        crop_action = CropAction((0.1, 0.1, 0.5, 0.5))
        gain = estimator.estimate_gain(crop_action, state)
        self.assertGreater(gain, 0)


class TestPDBDual(unittest.TestCase):
    """Test dual variable management and optimal stopping"""
    
    def setUp(self):
        self.config = {
            "visual_budget": 100,
            "text_budget": 50,
            "lambda_v_init": 0.02,
            "lambda_t_init": 0.01,
            "lambda_v_lr": 0.001,
            "lambda_t_lr": 0.001,
            "stopping_threshold": 0.0,
            "cost_per_patch": 1,
            "cost_crop_fixed": 32,
            "cost_draw_per_primitive": 4,
            "cost_per_text_token": 1,
            "gain_estimators": ["entropy"],
            "gain_weights": [1.0],
            "gain_temperature": 1.0
        }
    
    def test_dual_variable_update(self):
        """Test dual variable updates"""
        manager = DualVariableManager(self.config)
        
        initial_lambda_v, initial_lambda_t = manager.get_multipliers()
        self.assertEqual(initial_lambda_v, 0.02)
        self.assertEqual(initial_lambda_t, 0.01)
        
        # Update with overbudget usage
        manager.update(visual_cost=150, text_cost=60)
        
        new_lambda_v, new_lambda_t = manager.get_multipliers()
        # Lambdas should increase when over budget
        self.assertGreater(new_lambda_v, initial_lambda_v)
        self.assertGreater(new_lambda_t, initial_lambda_t)
        
        # Update with underbudget usage
        manager.update(visual_cost=50, text_cost=20)
        
        final_lambda_v, final_lambda_t = manager.get_multipliers()
        # Lambdas should decrease when under budget
        self.assertLess(final_lambda_v, new_lambda_v)
        self.assertLess(final_lambda_t, new_lambda_t)
    
    def test_optimal_stopping(self):
        """Test optimal stopping decision"""
        # Create mock components
        cost_model = CostModel(self.config)
        
        # Create a simple gain estimator
        gain_estimator = MagicMock()
        gain_estimator.forward = MagicMock(return_value=torch.tensor(0.5))
        
        dual_manager = DualVariableManager(self.config)
        
        stopping = OptimalStopping(
            self.config, cost_model, gain_estimator, dual_manager
        )
        
        # Test with various actions
        actions = [
            AnswerAction(),
            TextAction(10),
            CropAction((0.1, 0.1, 0.5, 0.5)),
            StopAction()
        ]
        
        state_embedding = torch.randn(256)
        state_dict = {"task_type": "general"}
        budget_tracker = BudgetTracker(100, 50)
        
        should_stop, best_action, info = stopping.should_stop(
            actions, state_embedding, state_dict, None, budget_tracker
        )
        
        # Should have made a decision
        self.assertIsInstance(should_stop, bool)
        if not should_stop:
            self.assertIsNotNone(best_action)
        
        # Info should contain decision details
        self.assertIn("action_values", info)
        self.assertIn("best_value", info)
        self.assertIn("lambda_v", info)
        self.assertIn("lambda_t", info)
    
    def test_pdb_controller(self):
        """Test the main PDB controller"""
        controller = PDBController(self.config)
        
        # Initialize episode
        tracker = controller.initialize_episode()
        self.assertIsInstance(tracker, BudgetTracker)
        
        # Create test actions
        actions = [
            AnswerAction(),
            TextAction(10),
            StopAction()
        ]
        
        state_embedding = torch.randn(256)
        state_dict = {"task_type": "general", "answer_entropy": 0.8}
        
        # Select action
        with patch.object(controller.gain_estimator, 'forward', return_value=torch.tensor(0.5)):
            action, info = controller.select_action(
                actions, state_embedding, state_dict, tracker
            )
        
        # Should return an action or None
        if action is not None:
            self.assertIn(action.action_type, [ActionType.ANS, ActionType.TEXT, ActionType.STOP])
        
        # End episode
        controller.end_episode(tracker)
        
        # Get statistics
        stats = controller.get_statistics()
        self.assertIn("dual_stats", stats)
        self.assertIn("config", stats)


class TestPDBIntegration(unittest.TestCase):
    """Test integration with training pipeline"""
    
    def test_import_modules(self):
        """Test that all modules can be imported"""
        try:
            from verl.utils.pdb_actions import ActionSpace
            from verl.utils.pdb_costs import CostModel
            from verl.utils.pdb_gains import CombinedGainEstimator
            from verl.utils.pdb_dual import PDBController
            
            # All imports successful
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import PDB modules: {e}")
    
    def test_config_structure(self):
        """Test that config structure is valid"""
        from verl.trainer.config import AlgorithmConfig
        
        config = AlgorithmConfig()
        
        # Check that pdb_config exists
        self.assertTrue(hasattr(config, 'pdb_config'))
        self.assertIsInstance(config.pdb_config, dict)
        
        # Check required keys
        required_keys = [
            "visual_budget", "text_budget", 
            "lambda_v_init", "lambda_t_init"
        ]
        for key in required_keys:
            self.assertIn(key, config.pdb_config)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPDBActions))
    suite.addTests(loader.loadTestsFromTestCase(TestPDBCosts))
    suite.addTests(loader.loadTestsFromTestCase(TestPDBGains))
    suite.addTests(loader.loadTestsFromTestCase(TestPDBDual))
    suite.addTests(loader.loadTestsFromTestCase(TestPDBIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)