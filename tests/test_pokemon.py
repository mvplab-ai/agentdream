"""The Claude-Plays-Pokémon case: a wrong location note is corrected from the logs, with citations."""

import json
import re
import shutil

from agentdream import dream

from conftest import EXAMPLES, final_contents, hash_tree, ops_of, write, write_jsonl


def _pokemon_logs(tmp_path):
    s = tmp_path / "logs"
    write_jsonl(s / "run1.jsonl", [
        {"role": "user", "content": "Heal your team at the Viridian City PokéCenter."},
        {"role": "assistant", "content": "According to my notes the PokéCenter is north of the gym. Heading north."},
        {"role": "assistant", "content": "Walked north of the gym. No PokéCenter here, just trees."},
        {"role": "assistant", "content": "Still no PokéCenter north of the gym. Going back."},
    ])
    write_jsonl(s / "run2.jsonl", [
        {"role": "assistant", "content": "Trying north of the gym again for the PokéCenter."},
        {"role": "assistant", "content": "Walked north of the gym again. No PokéCenter here."},
    ])
    write_jsonl(s / "run3.jsonl", [
        {"role": "assistant", "content": "Let me explore south of the gym instead."},
        {"role": "assistant", "content": "Found the PokéCenter! It is south of the gym."},
        {"role": "assistant", "content": "Healed at the PokéCenter, right south of the gym."},
    ])
    return s


def test_pokecenter_note_corrected_with_evidence(tmp_path):
    mem = write(tmp_path / "mem" / "MEMORY.md",
                "# Notes\n\n- The PokéCenter in Viridian City is north of the gym.\n- Pikachu knows Thunderbolt.\n")
    logs = _pokemon_logs(tmp_path)
    job = dream(mem.parent, [logs])
    assert job.status == "completed", job.error
    corrections = ops_of(job, "correct")
    assert len(corrections) == 1
    op = corrections[0]
    assert op.targets == ["MEMORY.md#1"]
    assert "south of the gym" in op.after[0]
    assert "north" not in op.after[0].lower()
    # evidence cites transcript file + line and quotes it
    cites = [e.citation for e in op.evidence]
    assert any(re.search(r"run3\.jsonl:\d+$", c) for c in cites), cites
    assert any(re.search(r"run[12]\.jsonl:\d+$", c) and e.stance == "refutes" for c, e in zip(cites, op.evidence))
    for ev in op.evidence:
        line_no = int(ev.citation.rsplit(":", 1)[1])
        line = open(ev.source_path, encoding="utf-8").read().splitlines()[line_no - 1]
        assert ev.quote.split(".")[0][:20] in json.loads(line)["content"]
    # the new store contains the corrected note; the other note is untouched
    contents = final_contents(job)
    assert "The PokéCenter in Viridian City is south of the gym." in contents
    assert "Pikachu knows Thunderbolt." in contents
    # report mentions it in the Corrected section with the citation
    md = job.report_md
    corrected_section = md.split("## Corrected")[1].split("## New insights")[0]
    assert "south of the gym" in corrected_section and "run3.jsonl:" in corrected_section
    assert job.report["sections"]["corrected"] == [op.id]


def test_correction_is_general_not_hardcoded(tmp_path):
    """Different subject, direction and wording -> same mechanism."""
    mem = write(tmp_path / "mem" / "notes.md", "- The ladder in Mt. Moon cave is east of the entrance.\n")
    s = tmp_path / "s"
    write(s / "a.txt", "Assistant: Went east of the entrance. Couldn't find the ladder anywhere.\n"
                       "Assistant: No ladder here east of the entrance either.\n")
    write(s / "b.txt", "Assistant: Turns out the ladder is west of the entrance, behind a boulder.\n"
                       "Assistant: Reached the ladder west of the entrance and climbed down.\n")
    job = dream(mem.parent, [s])
    (op,) = ops_of(job, "correct")
    assert op.after == ["The ladder in Mt. Moon cave is west of the entrance."]
    assert {e.citation.rsplit("/", 1)[-1] for e in op.evidence} >= {"b.txt:1"}


def test_number_fact_corrected_from_transcripts(tmp_path):
    mem = write(tmp_path / "m" / "facts.md", "- The dev server runs on port 3000.\n")
    s = tmp_path / "s"
    write_jsonl(s / "1.jsonl", [{"role": "assistant", "content": "Started it, but the dev server is actually on port 5173."},
                                {"role": "user", "content": "Right, the dev server port is 5173 now (Vite)."}])
    job = dream(mem.parent, [s])
    (op,) = ops_of(job, "correct")
    assert "5173" in op.after[0] and "3000" not in op.after[0]


def test_intent_alone_is_not_evidence(tmp_path):
    """The agent repeating its (wrong) belief must not 'confirm' or 'correct' anything."""
    mem = write(tmp_path / "m" / "MEMORY.md", "- The key is under the mat.\n- The vault is left of the stairs.\n")
    s = tmp_path / "s"
    write_jsonl(s / "1.jsonl", [{"role": "assistant", "content": "My notes say the vault is left of the stairs. Heading left."},
                                {"role": "assistant", "content": "I will try right of the stairs to find the vault later."}])
    job = dream(mem.parent, [s])
    assert ops_of(job, "correct") == []


def test_examples_dir_reproduces_pokemon_case(tmp_path):
    mem = tmp_path / "memory"
    shutil.copytree(EXAMPLES / "memory", mem)
    before = hash_tree(mem, EXAMPLES / "sessions")
    job = dream(mem, [EXAMPLES / "sessions"])
    assert hash_tree(mem, EXAMPLES / "sessions") == before
    corrected = [o for o in ops_of(job, "correct") if "PokéCenter" in o.before[0]]
    assert corrected and "south of the gym" in corrected[0].after[0]
    assert any("session-03.jsonl:2" in e.citation for e in corrected[0].evidence)
