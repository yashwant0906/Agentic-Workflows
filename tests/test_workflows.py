import ast
import importlib
import json
import unittest
from unittest.mock import MagicMock, patch

import common

agent_module = importlib.import_module("06_autonomous_agent")
chain_module = importlib.import_module("01_prompt_chaining")
routing_module = importlib.import_module("02_routing")
parallel_module = importlib.import_module("03_parallelization")
orch_module = importlib.import_module("04_orchestrator_workers")
eval_module = importlib.import_module("05_evaluator_optimizer")


class TestAutonomousAgentTools(unittest.TestCase):
    def test_safe_eval_basic(self):
        expr = "10 + 20 * 3"
        node = ast.parse(expr, mode="eval")
        self.assertEqual(agent_module._safe_eval(node), 70.0)

    def test_safe_eval_unary_and_float(self):
        expr = "+5 - -10"
        node = ast.parse(expr, mode="eval")
        self.assertEqual(agent_module._safe_eval(node), 15.0)

    def test_safe_eval_exponent_limit(self):
        node = ast.parse("2 ** 101", mode="eval")
        with self.assertRaises(ValueError):
            agent_module._safe_eval(node)

    def test_safe_eval_forbidden_syntax(self):
        with self.assertRaises(ValueError):
            agent_module._safe_eval(ast.parse("__import__('os').system('ls')", mode="eval"))
        with self.assertRaises(ValueError):
            agent_module._safe_eval(ast.parse("x + 1", mode="eval"))

    def test_run_tool_calculate(self):
        res = agent_module.run_tool("calculate", {"expression": "100 / 4"})
        self.assertEqual(res, "25.0")

    def test_run_tool_list_files(self):
        files = json.loads(agent_module.run_tool("list_files", {}))
        self.assertIn("sales.csv", files)

    def test_run_tool_read_csv(self):
        data = json.loads(agent_module.run_tool("read_csv", {"filename": "sales.csv", "limit": 2}))
        self.assertEqual(len(data), 2)
        self.assertIn("region", data[0])

    def test_run_tool_read_csv_security(self):
        with self.assertRaises(ValueError):
            agent_module.run_tool("read_csv", {"filename": "../common.py"})
        with self.assertRaises(ValueError):
            agent_module.run_tool("read_csv", {"filename": "non_existent.csv"})


class TestWorkflowPatternsWithMocks(unittest.TestCase):
    def test_prompt_chaining_passes(self):
        with patch.object(chain_module, "call") as mock_call, \
             patch.object(chain_module, "call_json") as mock_json:
            mock_call.side_effect = ["Outline 1-2-3", "Draft text", "Final edited blog post"]
            mock_json.return_value = {"passes": True, "missing": []}
            res = chain_module.chain("Test topic")
            self.assertEqual(res, "Final edited blog post")

    def test_routing(self):
        with patch.object(routing_module, "call") as mock_call, \
             patch.object(routing_module, "call_json") as mock_json:
            mock_json.return_value = {"route": "billing", "urgency": "high", "reason": "double charge"}
            mock_call.return_value = "Here is billing assistance."
            res = routing_module.route("I was billed twice")
            self.assertEqual(res, "Here is billing assistance.")
            self.assertIn(routing_module.ROUTES["billing"], mock_call.call_args[1]["system"])

    def test_parallelization(self):
        with patch.object(parallel_module, "call") as mock_call, \
             patch.object(parallel_module, "call_json") as mock_json:
            mock_call.return_value = "- Clean code\n- No bugs"
            mock_json.return_value = {"safe_to_merge": True, "reason": "All checks passed"}
            res = parallel_module.review("def foo(): pass", voters=2)
            self.assertIn("Verdict: APPROVE", res)

    def test_orchestrator_workers(self):
        with patch.object(orch_module, "call") as mock_call, \
             patch.object(orch_module, "call_json") as mock_json:
            mock_json.return_value = {
                "subtasks": [
                    {"title": "Cost", "instructions": "Evaluate cost"},
                    {"title": "Complexity", "instructions": "Evaluate complexity"},
                ]
            }
            mock_call.side_effect = ["Cost findings", "Complexity findings", "Final Research Brief"]
            res = orch_module.research("Evaluate K8s")
            self.assertEqual(res, "Final Research Brief")

    def test_evaluator_optimizer_pass_first_try(self):
        with patch.object(eval_module, "call") as mock_call, \
             patch.object(eval_module, "call_json") as mock_json:
            mock_call.return_value = "def parse(): pass"
            mock_json.return_value = {"verdict": "PASS", "feedback": []}
            res = eval_module.optimize("parse ISO durations")
            self.assertEqual(res, "def parse(): pass")

    def test_autonomous_agent_tool_loop(self):
        with patch.object(agent_module, "Conversation") as mock_conv_cls:
            mock_conv = MagicMock()
            mock_conv_cls.return_value = mock_conv

            # Step 1: Model asks to calculate 2 + 2
            reply_step1 = common.Reply(
                text="",
                tool_calls=[common.ToolCall(id="c1", name="calculate", input={"expression": "2 + 2"})],
            )
            # Step 2: Model finishes with answer
            reply_step2 = common.Reply(text="The result is 4.0", tool_calls=[])
            mock_conv.send.side_effect = [reply_step1, reply_step2]

            res = agent_module.agent("What is 2 + 2?")
            self.assertEqual(res, "The result is 4.0")
            mock_conv.add_tool_results.assert_called_once_with([("c1", "4.0", False)])


if __name__ == "__main__":
    unittest.main()
