import os
import unittest
from unittest.mock import patch

import common


class TestCommonConfig(unittest.TestCase):
    def setUp(self):
        common.reset_client_cache()

    def tearDown(self):
        common.reset_client_cache()

    @patch.dict(os.environ, {}, clear=True)
    def test_provider_not_configured(self):
        self.assertFalse(common.is_configured())
        with self.assertRaises(common.ConfigError):
            common.provider()

    @patch.dict(os.environ, {"LLM_PROVIDER": "invalid"}, clear=True)
    def test_provider_invalid_explicit(self):
        with self.assertRaises(common.ConfigError):
            common.provider()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True)
    def test_provider_auto_detect_anthropic(self):
        self.assertTrue(common.is_configured())
        self.assertEqual(common.provider(), "anthropic")
        self.assertEqual(common.model(), "claude-opus-5")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai-test"}, clear=True)
    def test_provider_auto_detect_openai(self):
        self.assertTrue(common.is_configured())
        self.assertEqual(common.provider(), "openai")
        self.assertEqual(common.model(), "gpt-5")

    @patch.dict(os.environ, {"LLM_BASE_URL": "http://localhost:11434/v1"}, clear=True)
    def test_provider_auto_detect_local(self):
        self.assertTrue(common.is_configured())
        self.assertEqual(common.provider(), "local")
        with self.assertRaises(common.ConfigError):
            common.model()

    @patch.dict(os.environ, {"LLM_BASE_URL": "http://localhost:11434/v1", "LLM_MODEL": "qwen3:1.7b"}, clear=True)
    def test_provider_local_with_model(self):
        self.assertEqual(common.provider(), "local")
        self.assertEqual(common.model(), "qwen3:1.7b")

    @patch.dict(os.environ, {"LLM_PROVIDER": "openai", "LLM_MODEL": "custom-gpt"}, clear=True)
    def test_explicit_provider_override(self):
        self.assertEqual(common.provider(), "openai")
        self.assertEqual(common.model(), "custom-gpt")


class TestCommonJsonHelpers(unittest.TestCase):
    def test_extract_json_direct(self):
        self.assertEqual(common._extract_json('{"key": "value"}'), {"key": "value"})
        self.assertEqual(common._extract_json('[1, 2, 3]'), [1, 2, 3])

    def test_extract_json_markdown_fenced(self):
        raw = "Here is the response:\n```json\n{\"status\": \"ok\", \"count\": 42}\n```\nHope that helps!"
        self.assertEqual(common._extract_json(raw), {"status": "ok", "count": 42})

    def test_extract_json_markdown_without_tag(self):
        raw = "```\n[{\"id\": 1}]\n```"
        self.assertEqual(common._extract_json(raw), [{"id": 1}])

    def test_extract_json_invalid(self):
        with self.assertRaises(ValueError):
            common._extract_json("no json here")
        with self.assertRaises(ValueError):
            common._extract_json("{broken json")

    def test_validate_types_and_enum(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "role": {"type": "string", "enum": ["admin", "user"]},
            },
            "required": ["name", "age"],
        }
        valid_data = {"name": "Alice", "age": 30, "active": True, "tags": ["lead"], "role": "admin"}
        # Should not raise
        common._validate(valid_data, schema)

        # Missing required
        with self.assertRaises(ValueError):
            common._validate({"age": 30}, schema)

        # Invalid type
        with self.assertRaises(ValueError):
            common._validate({"name": 123, "age": 30}, schema)

        # Invalid enum
        with self.assertRaises(ValueError):
            common._validate({"name": "Alice", "age": 30, "role": "superadmin"}, schema)

        # Boolean passed for integer/number
        with self.assertRaises(ValueError):
            common._validate({"name": "Alice", "age": True}, schema)

    def test_validate_union_type(self):
        schema = {
            "type": "object",
            "properties": {
                "note": {"type": ["string", "null"]},
            },
        }
        common._validate({"note": "hello"}, schema)
        common._validate({"note": None}, schema)
        with self.assertRaises(ValueError):
            common._validate({"note": 123}, schema)


class TestConversation(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    def test_openai_conversation_tool_results(self):
        common.reset_client_cache()
        conv = common.Conversation(system="system prompt")
        conv.add_user("hello")
        self.assertEqual(conv.messages[0], {"role": "system", "content": "system prompt"})
        self.assertEqual(conv.messages[1], {"role": "user", "content": "hello"})

        conv.add_tool_results([("call_1", "result_1", False)])
        self.assertEqual(conv.messages[2], {"role": "tool", "tool_call_id": "call_1", "content": "result_1"})

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True)
    def test_anthropic_conversation_tool_results(self):
        common.reset_client_cache()
        conv = common.Conversation(system="system prompt")
        self.assertEqual(conv.messages, [])
        conv.add_user("hello")
        conv.add_tool_results([("call_1", "res", False)])
        self.assertEqual(conv.messages[1]["role"], "user")
        self.assertEqual(
            conv.messages[1]["content"],
            [{"type": "tool_result", "tool_use_id": "call_1", "content": "res", "is_error": False}],
        )


if __name__ == "__main__":
    unittest.main()
