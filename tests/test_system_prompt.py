"""System-prompt support in the conversation session (both backends)."""

from __future__ import annotations

from dipai.inference import ConversationSession


class TestSystemPrompt:
    def test_empty_by_default(self):
        s = ConversationSession()
        assert s.system_prompt == ""

    def test_plain_prompt_unchanged_without_system(self):
        s = ConversationSession()
        s.add("user", "hi")
        p = s.as_prompt()
        assert not p.startswith("System:")
        assert p.endswith("\n\nAssistant: ")

    def test_system_prefix_in_as_prompt(self):
        s = ConversationSession()
        s.system_prompt = "Be brief."
        s.add("user", "hi")
        p = s.as_prompt()
        assert p.startswith("System: Be brief.")
        assert "User: hi" in p
        assert p.endswith("\n\nAssistant: ")

    def test_system_only_no_turns(self):
        s = ConversationSession()
        s.system_prompt = "Be brief."
        assert s.as_prompt() == "System: Be brief."

    def test_whitespace_system_ignored(self):
        s = ConversationSession()
        s.system_prompt = "   "
        s.add("user", "hi")
        assert not s.as_prompt().startswith("System:")

    def test_as_messages_includes_system_first(self):
        s = ConversationSession()
        s.system_prompt = "You are terse."
        s.add("user", "q")
        s.add("assistant", "a")
        msgs = s.as_messages()
        assert msgs[0] == {"role": "system", "content": "You are terse."}
        assert msgs[1] == {"role": "user", "content": "q"}
        assert msgs[2] == {"role": "assistant", "content": "a"}

    def test_as_messages_omits_system_when_empty(self):
        s = ConversationSession()
        s.add("user", "q")
        assert all(m["role"] != "system" for m in s.as_messages())

    def test_clear_keeps_system_prompt(self):
        s = ConversationSession()
        s.system_prompt = "Keep me."
        s.add("user", "x")
        s.clear()
        assert s.system_prompt == "Keep me."
        assert len(s) == 0

    def test_history_trim_never_drops_system(self):
        s = ConversationSession(max_turns=2)
        s.system_prompt = "sys"
        for i in range(5):
            s.add("user", f"u{i}")
            s.add("assistant", f"a{i}")
        assert s.system_prompt == "sys"
        assert len(s) == 2
        assert s.as_prompt().startswith("System: sys")
