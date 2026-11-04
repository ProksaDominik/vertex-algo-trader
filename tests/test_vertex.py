"""Vertex, the voice butler (2026-10-09): the stats it answers from, the proof check, settings, the wake rule."""
from __future__ import annotations

import json
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vertex_voice"))

from metatraderbot import vertex_stats as vs  # noqa: E402
import vertex_voice as vv  # noqa: E402

NOW = datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)          # Fri 9 Oct 11:00 New York


def _write(d: Path, name: str, obj) -> None:
    p = d / name
    if isinstance(obj, list):
        p.write_text("\n".join(json.dumps(x) for x in obj) + "\n", encoding="utf-8")
    else:
        p.write_text(json.dumps(obj), encoding="utf-8")


def _data(tmp: Path) -> Path:
    _write(tmp, "state_lucid_futures.json", {
        "attempts": 3, "passes": 2, "fails": 1, "race_day": "2026-10-09", "race_attempt": 4,
        "attempt": {"realized_usd": 0.0, "tickets": {}},
        "history": [{"outcome": "pass", "resolved_ts": "2026-10-08T14:00:00+00:00"},      # Thu
                    {"outcome": "fail", "resolved_ts": "2026-10-08T20:00:00+00:00"},      # 16:00 ET Thu
                    {"outcome": "pass", "resolved_ts": "2026-10-09T13:00:00+00:00"}]})    # Fri
    _write(tmp, "lucid_futures_trades.jsonl", [
        {"attempt": 1, "entry_price": 100, "exit_price": 175.5, "lots": 2, "pnl_usd": 3013.0, "reason": "tp",
         "opened_ts": "2026-10-07T23:00:00+00:00", "closed_ts": "2026-10-08T14:00:00+00:00"},
        {"attempt": 2, "entry_price": 100, "exit_price": 50, "lots": 2, "pnl_usd": -2050.0, "reason": "closed_by_bot",
         "opened_ts": "2026-10-08T14:01:00+00:00", "closed_ts": "2026-10-08T19:00:00+00:00"},
        {"attempt": 3, "entry_price": 100, "exit_price": 175.5, "lots": 2, "pnl_usd": 3013.0, "reason": "pass_while_away",
         "opened_ts": "2026-10-08T23:00:00+00:00", "closed_ts": "2026-10-09T13:00:00+00:00"}])
    _write(tmp, "state_lucid_futures_funded.json", {
        "account": "DEMO1", "runs_total": 2, "runs_blown": 1, "need_usd": 1569.75, "floating_usd": 0.0,
        "run": {"realized_usd": 1046.5, "paid_usd": 0.0, "day_start_realized": 0.0, "payouts": [], "tickets": {}},
        "history": [{"run": 1, "outcome": "blown", "end": "2026-10-07T09:00:00+00:00",
                     "payout_list": [{"amount_usd": 500.0, "to_you_usd": 450.0, "ts": "2026-10-06T21:00:00+00:00"}]}]})
    _write(tmp, "lucid_futures_funded_trades.jsonl", [
        {"run": 2, "entry_price": 100, "exit_price": 152.5, "lots": 1, "pnl_usd": 1046.5, "reason": "tp",
         "opened_ts": "2026-10-08T23:00:00+00:00", "closed_ts": "2026-10-09T02:00:00+00:00"}])
    _write(tmp, "prediction_lab.json", {"days": 4, "summary": {"deployed": {"raced": 4, "wins": 1, "status": "watching"}}})
    return tmp


def test_periods_split_by_trading_day(tmp_path):
    s = vs.summary(_data(tmp_path), NOW)
    t = s["trial"]["periods"]
    assert s["trading_day"] == "2026-10-09"
    assert t["today"]["evaluations_passed"] == 1 and t["today"]["races"] == 1
    assert t["yesterday"]["evaluations_passed"] == 1 and t["yesterday"]["evaluations_failed"] == 1
    assert t["yesterday"]["hit_loss_line"] == 1 and t["yesterday"]["pnl_usd"] == 963.0
    assert t["all_time"]["pass_rate"] == round(2 / 3, 3)
    f = s["funded"]
    assert f["accounts"][0]["balance_usd"] == 51_046.5 and f["periods"]["today"]["reached_target"] == 1
    assert f["periods"]["this_month"]["payouts"] == 1 and f["periods"]["this_month"]["to_you_usd"] == 450.0
    assert f["periods"]["this_month"]["accounts_blown"] == 1
    assert s["next_race"]["new_york"].startswith("Sun 11 Oct 19:00")    # raced today -> after the weekend


def test_events_after_a_time(tmp_path):
    ev = vs.events(_data(tmp_path), datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc))
    texts = [e["text"] for e in ev]
    assert any("reached its target" in t for t in texts) and any("PASSED" in t for t in texts)
    assert any(t.startswith("Funded race on DEMO1 won") for t in texts)
    assert not any("Payout" in t for t in texts)                          # the payout was on 6 Oct


def test_proof_is_looked_up_in_the_records(tmp_path):
    s = vs.summary(_data(tmp_path), NOW)
    rows = vv.resolve_proof(s, [{"label": "passed today", "path": "trial.periods.today.evaluations_passed"},
                                {"label": "balance", "path": "funded.accounts.0.balance_usd"},
                                {"label": "pnl", "path": "trial.periods.yesterday.pnl_usd"},
                                {"label": "made up", "path": "trial.periods.today.profit_forecast"}])
    assert rows[0] == ("passed today", "1", True)
    assert rows[1] == ("balance", "$51,046.50", True)
    assert rows[2] == ("pnl", "$963", True)
    assert rows[3] == ("made up", "not in the records", False)
    assert vs.lookup(s, "funded.accounts.5.balance_usd") is None


def test_settings_file_and_environment(tmp_path, monkeypatch):
    p = tmp_path / "settings.env"
    p.write_text('# comment\nVERTEX_SERVER=vertex-server\nANTHROPIC_API_KEY="sk-ant-x"\nVERTEX_ANNOUNCE=0\n',
                 encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = vv.load_settings(p)
    assert s["VERTEX_SERVER"] == "vertex-server" and s["ANTHROPIC_API_KEY"] == "sk-ant-x"
    assert s["VERTEX_ANNOUNCE"] == "0" and s["VERTEX_VOICE"] == "en-GB-RyanNeural"


def test_wake_needs_both_words_heard_surely():
    me = SimpleNamespace(MIN_CONF=0.75)
    sure = {"result": [{"word": "hey", "conf": 0.98, "end": 0.4}, {"word": "vertex", "conf": 0.95, "end": 0.9}]}
    unsure = {"result": [{"word": "hey", "conf": 0.98, "end": 0.4}, {"word": "vertex", "conf": 0.42, "end": 0.9}]}
    alone = {"result": [{"word": "vertex", "conf": 1.0, "end": 0.5}]}
    assert vv.Ears.wake_in(me, sure) == 0.9
    assert vv.Ears.wake_in(me, unsure) is None and vv.Ears.wake_in(me, alone) is None


def test_the_answer_format_allows_only_known_tables():
    assert set(vv.ANSWER_SCHEMA["properties"]["show"]["items"]["enum"]) == set(vv.TABLES)
    assert "Never invent" in vv.system_prompt("")
    from metatraderbot import app_vertex as av
    assert av.TABLES == vv.TABLES                          # the dashboard draws the tables the app names


# --- Vertex inside the dashboard (2026-10-09: "keep it all in that dashboard") ------------------------------

def test_log_state_inbox_and_screen(tmp_path):
    vs.record_vertex_event(tmp_path, {"kind": "state", "viewer": "Work Laptop!", "state": "listening",
                                      "voice": False})
    live = vs.vertex_live(tmp_path)
    assert live["WorkLaptop"]["state"] == "listening" and live["WorkLaptop"]["online"]
    assert live["WorkLaptop"]["voice"] is False
    later = datetime.now(timezone.utc).replace(year=2099)
    assert vs.vertex_live(tmp_path, later)["WorkLaptop"]["online"] is False         # silent for too long
    ask = vs.record_vertex_event(tmp_path, {"kind": "ask", "viewer": "WorkLaptop", "text": "How did we do?"})
    vs.record_vertex_event(tmp_path, {"kind": "ask", "viewer": "HOME", "text": "not for the work laptop"})
    box = vs.vertex_inbox(tmp_path, "WorkLaptop", "")
    assert [e["text"] for e in box] == ["How did we do?"]
    assert vs.vertex_inbox(tmp_path, "WorkLaptop", ask["ts"]) == []                 # already picked up
    vs.record_screen(tmp_path, "WorkLaptop", "FUNDED")
    assert vs.vertex_screen(tmp_path, "WorkLaptop")["page"] == "funded"
    assert vs.vertex_screen(tmp_path, "Unknown") == {}


def test_both_laptops_announce_but_the_log_keeps_it_once(tmp_path):
    for laptop in ("HOME", "WORK"):
        vs.record_vertex_event(tmp_path, {"kind": "announce", "viewer": laptop, "text": "An evaluation PASSED.",
                                          "key": "2026-10-09T13:00:00+00:00 An evaluation PASSED."})
    assert [e["kind"] for e in vs.vertex_feed(tmp_path)] == ["announce"]


def test_an_answer_is_stored_with_its_proof_and_screen_move(tmp_path):
    e = vs.record_vertex_event(tmp_path, {
        "kind": "exchange", "viewer": "HOME", "question": "Show me the payouts", "speech": "Here they are.",
        "proof": [{"label": "Payouts", "path": "funded.periods.this_month.payouts", "value": "1", "found": True}],
        "show": ["funded.payouts"], "ui": {"page": "funded", "tab": "Payouts & accounts", "bad": {"x": 1}}})
    assert e["proof"][0]["found"] and e["ui"] == {"page": "funded", "tab": "Payouts & accounts"}
    try:
        vs.record_vertex_event(tmp_path, {"kind": "place_order"})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown kinds must be refused")


def test_quick_commands_need_no_claude():
    q = vv.quick_command
    assert q("Show me the funded accounts")["ui"] == {"page": "funded"}
    assert q("open the payouts tab")["ui"] == {"page": "funded", "tab": "Payouts & accounts"}
    assert q("go to the trial trade charts")["ui"] == {"page": "trial", "tab": "Trade charts"}
    assert q("open trade charts", "funded")["ui"] == {"page": "funded", "tab": "Trade charts"}   # the page on screen
    assert q("scroll down")["ui"] == {"scroll": "down"} and q("go to the top")["ui"] == {"scroll": "top"}
    assert q("Mute.")["flags"] == {"voice": False} and q("voice on")["flags"] == {"voice": True}
    assert q("announcements off")["flags"] == {"announce": False}
    assert q("How much did we make this month?") is None                       # a question: Claude
    assert q("show me how the funded account did today") is None


def test_claude_screen_moves_are_cleaned():
    assert vv.clean_ui({"page": "fleet", "tab": "none", "account": "", "fleet_period": "Last 3 months",
                        "fleet_start": 3, "fleet_up_to": 0, "scroll": "none"}) == \
        {"page": "fleet", "fleet_period": "Last 3 months", "fleet_start": 3}
    assert vv.clean_ui({"page": "trial", "tab": "Payouts & accounts"}) == {"page": "trial"}   # not on that page
    assert vv.clean_ui({"page": "none", "fleet_start": 9, "scroll": "sideways"}) == {}
    ui = vv.ANSWER_SCHEMA["properties"]["ui"]
    assert set(ui["required"]) == set(ui["properties"]) and ui["additionalProperties"] is False


def test_the_stats_service_takes_reports_and_gives_the_inbox(tmp_path, monkeypatch):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    sys.path.insert(0, str(ROOT / "scripts"))
    import vertex_api
    monkeypatch.setattr(vertex_api, "DATA", tmp_path)
    monkeypatch.delenv("VERTEX_API_TOKEN", raising=False)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), vertex_api.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        req = urllib.request.Request(base + "/vertex/event", data=json.dumps(
            {"kind": "state", "viewer": "HOME", "state": "thinking"}).encode(), headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req, timeout=5).read())["ok"]
        assert vs.vertex_live(tmp_path)["HOME"]["state"] == "thinking"
        vs.record_vertex_event(tmp_path, {"kind": "ask", "viewer": "HOME", "text": "hello"})
        vs.record_screen(tmp_path, "HOME", "TRIAL")
        box = json.loads(urllib.request.urlopen(base + "/vertex/inbox?viewer=HOME&after=", timeout=5).read())
        assert [q["text"] for q in box["questions"]] == ["hello"] and box["screen"]["page"] == "trial"
        bad = urllib.request.Request(base + "/vertex/event", data=b'{"kind": "nope"}')
        try:
            urllib.request.urlopen(bad, timeout=5)
            raise AssertionError("a bad report must be refused")
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        srv.shutdown()


def test_dashboard_pieces_escape_text_and_carry_the_command():
    from zoneinfo import ZoneInfo
    from metatraderbot import app_vertex as av
    tz = ZoneInfo("Europe/Prague")
    e = {"kind": "exchange", "ts": "2026-10-09T15:00:00+00:00", "viewer": "HOME", "source": "voice",
         "question": "<script>alert(1)</script>", "speech": "Fine & dandy",
         "proof": [{"label": "Balance", "value": "$51,046.50", "found": True, "path": "funded.accounts.0.balance_usd"}],
         "ui": {"page": "funded", "tab": "Payouts & accounts"}}
    h = av.entry_html(e, tz, latest=True)
    assert "<script>" not in h and "&lt;script&gt;" in h and "Fine &amp; dandy" in h
    assert "$51,046.50" in h and "funded.accounts.0.balance_usd" in h and "Funded page · Payouts &amp; accounts" in h
    card = av.dock_html(e, "speaking", "HOME", tz)
    assert "Speaking" in card and "Balance" in card
    js = av.command_script({"page": "Funded", "tab": "Payouts & accounts", "ignored": 1}, "abc123")
    assert '\\"page\\": \\"Funded\\"' in js and "ignored" not in js and "abc123" in js
    assert "</script>" not in js[js.index("<script>") + 8:-len("</script>")]


def test_speech_is_split_so_the_first_sentence_plays_sooner():
    assert vv.sentences("Not a good day so far. The trial bot lost one race. Yes. The funded account was blown.") == \
        ["Not a good day so far.", "The trial bot lost one race.", "Yes. The funded account was blown."]
    assert vv.sentences("One moment.") == ["One moment."]


def test_commands_with_details_go_to_claude_and_misheard_trial_still_opens_it():
    assert vv.quick_command("Show the fleet for the last 3 months") is None          # the period needs Claude
    assert vv.quick_command("Show me the fleet with 3 accounts") is None
    assert vv.quick_command("Open the fleet")["ui"] == {"page": "fleet"}
    assert vv.quick_command("Show me the travel page")["ui"] == {"page": "trial"}    # Whisper heard "travel"


# --- 2026-10-16: the PC butler and the patient ears ------------------------------------------------------------------

import pc_control as pc  # noqa: E402


def test_music_commands_need_no_claude():
    def act(q):
        return vv.quick_command(q)["actions"]
    assert act("Pause") == [{"do": "media", "text": "pause", "number": 0.0}]
    assert act("Stop the music.") == [{"do": "media", "text": "pause", "number": 0.0}]
    assert act("Next song please") == [{"do": "media", "text": "next", "number": 0.0}]
    assert act("Skip this one") == [{"do": "media", "text": "next", "number": 0.0}]
    assert act("Play the previous song") == [{"do": "media", "text": "previous", "number": 0.0}]
    assert act("Louder") == [{"do": "volume", "text": "up", "number": 0.0}]
    assert act("Turn it down a bit") == [{"do": "volume", "text": "down", "number": 0.0}]
    assert act("Set the volume to 40%.") == [{"do": "volume", "text": "", "number": 40.0}]
    assert vv.quick_command("What's playing?")["now_playing"] is True
    assert vv.quick_command("Play some romantic jazz") is None                     # new music: Claude picks it
    assert vv.quick_command("Stop talking")["flags"] == {"voice": False}           # Vertex's own voice, as before


def test_a_sentence_that_stops_half_way_is_not_finished():
    for t in ("Put on some music and", "Open the", "I want to", "Remind me in twenty minutes to take the pizza,",
              "Turn the volume up to...", "um"):
        assert vv.unfinished(t), t
    for t in ("Put on some music.", "How did we do today?", "Open Netflix", "Turn the lights on", "Thank you"):
        assert not vv.unfinished(t), t


class _FakeEars(vv.Ears):
    """The ears without a microphone: 0.1 s blocks of speech ("1") or quiet ("0"), and a fixed transcript."""

    def __init__(self, pattern: str, words: str):
        import queue

        import numpy as np
        self.np, self.pause, self.level, self.after_wake = np, 1.1, [], b""
        self._vad_win = np.zeros(self.VAD_WIN, dtype=np.float32)
        self.words = words
        self.q = queue.Queue()
        for ch in pattern:
            self.q.put((b"\x01\x00" if ch == "1" else b"\x00\x00") * self.BLOCK)

    def _speech(self, b):
        return b[:1] == b"\x01"

    def transcribe(self, audio):
        return self.words

    def _interim(self, chunks):
        return {"text": self.words, "done": True}


def test_vertex_waits_for_the_end_of_a_sentence_before_answering():
    # 2 s of speech, 1.5 s of quiet, 1 s of speech: a finished sentence ends after 1.1 s of quiet ...
    ears = _FakeEars("1" * 20 + "0" * 15 + "1" * 10 + "0" * 40, "Play some jazz.")
    audio, text = ears.record(start_timeout=1)
    assert len(audio) == (20 + 11) * 1_600 and text == "Play some jazz."
    # ... but the same pause after "... and" is a breath: Vertex keeps listening and gets the rest
    ears = _FakeEars("1" * 20 + "0" * 15 + "1" * 10 + "0" * 40, "Play some jazz and")
    audio, _ = ears.record(start_timeout=1)
    assert len(audio) == (20 + 15 + 10 + 27) * 1_600
    assert _FakeEars("0" * 10, "").record(start_timeout=0.2) is None                # nobody spoke


def test_commands_that_could_hurt_the_bots_are_refused_whatever_the_owner_says():
    assert pc.blocked("Stop-Process -Name python -Force")
    assert pc.blocked("taskkill /IM pythonw.exe /F")
    assert pc.blocked("Remove-Item 'C:\\Users\\Proksha\\OneDrive\\ruflo\\metatraderbot\\data' -Recurse")
    assert pc.blocked("Get-Content $env:LOCALAPPDATA\\VertexVoice\\settings.env")
    assert pc.blocked("Disable-ScheduledTask -TaskName 'Vertex Watchdog'")
    assert pc.blocked("Format-Volume -DriveLetter D")
    assert pc.blocked("Stop-Computer")                                              # the shut-down action asks
    assert not pc.blocked("Get-PSDrive C | Select-Object Used, Free")
    assert not pc.blocked("Set-ItemProperty HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize "
                          "-Name AppsUseLightTheme -Value 0")


def test_actions_are_cleaned_and_described_and_a_yes_is_a_yes():
    acts = pc.clean_actions([{"do": "play_music", "text": " soft jazz ", "number": 1},
                             {"do": "format_disk", "text": "", "number": 0}, "junk",
                             {"do": "volume", "text": "", "number": 30}])
    assert acts == [{"do": "play_music", "text": "soft jazz", "number": 1.0},
                    {"do": "volume", "text": "", "number": 30.0}]
    assert [pc.describe(a) for a in acts] == ['music: "soft jazz"', "volume 30%"]
    assert set(pc.NEEDS_YES) == {"sleep", "shutdown", "restart"}
    assert pc.is_yes("Yes.") and pc.is_yes("Yeah, go ahead") and pc.is_yes("Sure")
    assert not pc.is_yes("No.") and not pc.is_yes("Yes, no wait") and not pc.is_yes("") and not pc.is_yes("maybe")


def test_spoken_app_names_find_the_app():
    names = ["Google Chrome", "Spotify", "Netflix", "Obsidian", "Word 2013", "Microsoft Edge", "Steam",
             "Steam Support Center", "Settings"]
    assert pc.app_match("chrome", names) == "Google Chrome"
    assert pc.app_match("the Spotify app", names) == "Spotify"
    assert pc.app_match("word", names) == "Word 2013"
    assert pc.app_match("steam", names) == "Steam"
    assert pc.app_match("netflics", names) == "Netflix"
    assert pc.app_match("photoshop", names) is None


def test_timers_and_memory_are_kept_on_disk(tmp_path):
    h = pc.Hands(tmp_path / "Vertex remembers.md", tmp_path / "timers.json", load_apps=False)
    assert h.do({"do": "remember", "text": "My girlfriend's name is Anna.", "number": 0}) == ""
    assert h.do({"do": "remember", "text": "Romantic music means slow jazz", "number": 0}) == ""
    assert "- My girlfriend's name is Anna" in h.memory_text() and "slow jazz" in h.memory_text()
    assert h.do({"do": "forget", "text": "girlfriend", "number": 0}) == ""
    assert "Anna" not in h.memory_text() and "slow jazz" in h.memory_text()
    assert (tmp_path / "Vertex remembers.md").read_text(encoding="utf-8").startswith("# Vertex remembers")
    assert h.do({"do": "timer", "text": "Take the pizza out.", "number": 0.0}) != ""        # no time: not set
    h.do({"do": "timer", "text": "Take the pizza out.", "number": 1200})
    again = pc.Hands(tmp_path / "Vertex remembers.md", tmp_path / "timers.json", load_apps=False)
    assert [t["text"] for t in again.timers] == ["Take the pizza out."]                   # survives a restart
    h.timers.append({"due": "2026-01-01T00:00:00", "text": "Old one."})
    assert h.due_timers() == ["Old one."] and [t["text"] for t in h.timers] == ["Take the pizza out."]


def test_the_answer_is_found_after_a_web_search():
    def T(s):
        return SimpleNamespace(type="text", text=s)
    content = [SimpleNamespace(type="server_tool_use"), SimpleNamespace(type="web_search_tool_result"),
               T('{"speech": "Tomorrow in Prague: twelve degrees'), T(' and rain.", "proof": [], "show": [], '),
               T('"ui": {}, "actions": []}')]
    assert vv.answer_json(content)["speech"] == "Tomorrow in Prague: twelve degrees and rain."
    assert vv.answer_json([T("plain words")])["speech"] == "plain words"
    assert vv.ANSWER_SCHEMA["properties"]["actions"]["items"]["properties"]["do"]["enum"] == list(pc.ACTIONS)
    assert "{city}" not in vv.system_prompt("", "Prague") and "Prague" in vv.system_prompt("", "Prague")


def test_what_vertex_did_is_stored_and_shown(tmp_path):
    from metatraderbot import app_vertex as av
    e = vs.record_vertex_event(tmp_path, {"kind": "exchange", "question": "romantic evening", "speech": "Of course.",
                                          "did": ['music: "soft romantic jazz"', "volume 30%"], "viewer": "laptop"})
    assert e["did"] == ['music: "soft romantic jazz"', "volume 30%"]
    html = av.entry_html(e, av.ZoneInfo("Europe/Prague"))
    assert "Done: <i>music: &quot;soft romantic jazz&quot; · volume 30%</i>" in html


# --- 2026-10-16: Spotify (the owner has Premium) ---------------------------------------------------------------------

import spotify as sp  # noqa: E402


class _FakeSpotify:
    """Spotify's Web API as a dict of answers; records every call."""

    def __init__(self, play_404_first=False):
        self.calls, self.play_404 = [], play_404_first

    def __call__(self, method, url, body):
        self.calls.append((method, url.replace(sp.API, ""), body))
        path = url.replace(sp.API, "").split("?")[0]
        if path == "/search" and "type=playlist" in url:
            return 200, {"playlists": {"items": [None, {"name": "love 3 songs", "uri": "spotify:playlist:tiny",
                                                         "tracks": {"total": 3}},
                                                  {"name": "Romantic R&B", "uri": "spotify:playlist:rnb",
                                                   "tracks": {"total": 80}}]}}
        if path == "/search" and "type=track" in url:
            return 200, {"tracks": {"items": [{"name": "Spaceship", "uri": "spotify:track:1",
                                               "artists": [{"name": "Viktor Sheen"}],
                                               "album": {"uri": "spotify:album:A"}}]}}
        if path == "/me":
            return 200, {"id": "owner1"}
        if path == "/search" and "type=artist" in url:
            return 200, {"artists": {"items": [{"name": "Viktor Sheen", "uri": "spotify:artist:9"}]}}
        if path == "/me/playlists":
            return 200, {"items": [{"name": "GYM 2026", "uri": "spotify:playlist:gym"},
                                   {"name": "chill vibes", "uri": "spotify:playlist:chill"}]}
        if path == "/me/tracks":
            return 200, {"items": [{"track": {"uri": f"spotify:track:L{i}"}} for i in range(5)]}
        if path == "/me/player/devices":
            return 200, {"devices": [{"id": "phone", "type": "Smartphone", "name": "iPhone", "is_active": True},
                                     {"id": "pc", "type": "Computer", "name": "PROKSHALNV", "is_active": False}]}
        if path == "/me/player/play" and self.play_404:
            self.play_404 = False
            return 404, {"error": {"status": 404, "message": "Player command failed: No active device found"}}
        return 204, {}


def _spotify(tmp_path, fake):
    (tmp_path / "tok.json").write_text(json.dumps({"access_token": "a", "refresh_token": "r", "expires_at": 9e12}))
    return sp.Spotify("client123", tmp_path / "tok.json", http=fake)


def test_spotify_sign_in_uses_pkce_without_a_secret():
    # RFC 7636 appendix B
    assert sp.challenge_for("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    v, c = sp.pkce_pair()
    assert 43 <= len(v) <= 128 and c == sp.challenge_for(v)
    assert sp.REDIRECT == "http://127.0.0.1:8520/callback"         # what the owner types in the Spotify app


def test_spotify_finds_songs_moods_artists_and_the_owners_playlists(tmp_path):
    s = _spotify(tmp_path, _FakeSpotify())
    assert s.ready()
    # a song = its album started at the song (a bare list of songs is accepted by Spotify but plays nothing)
    assert s.find("Spaceship Viktor Sheen", 0) == {"name": "Spaceship by Viktor Sheen",
                                                   "context_uri": "spotify:album:A", "offset_uri": "spotify:track:1"}
    mood = s.find("soft romantic R&B", 1)                           # a real playlist, not the 3-song one
    assert mood["context_uri"] == "spotify:playlist:rnb" and mood["shuffle"] is True
    assert s.find("Viktor Sheen", 2)["context_uri"] == "spotify:artist:9"
    assert s.find("my gym playlist", 4)["context_uri"] == "spotify:playlist:gym"
    assert s.find("chill", 4)["name"] == "chill vibes"
    assert s.find("party", 4) is None                               # not one of theirs
    assert s.find("", 5)["context_uri"] == "spotify:user:owner1:collection"          # Liked Songs, as a whole
    assert not sp.Spotify("client123", tmp_path / "none.json").ready()   # not signed in yet


def test_spotify_plays_on_this_pc_and_wakes_it_when_no_player_is_active(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPUTERNAME", "PROKSHALNV")
    fake = _FakeSpotify(play_404_first=True)
    s = _spotify(tmp_path, fake)
    assert s.play({"name": "Romantic R&B", "context_uri": "spotify:playlist:rnb", "shuffle": True}) == ""
    plays = [(m, u, b) for m, u, b in fake.calls if m == "PUT"]
    assert plays[0] == ("PUT", "/me/player/play?device_id=pc", {"context_uri": "spotify:playlist:rnb"})   # 404
    assert plays[1] == ("PUT", "/me/player", {"device_ids": ["pc"], "play": False})                      # wake
    assert plays[2] == ("PUT", "/me/player/play?device_id=pc", {"context_uri": "spotify:playlist:rnb"})
    assert plays[3][1].startswith("/me/player/shuffle?state=true")


def test_music_goes_to_spotify_when_connected_and_youtube_otherwise(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "find_youtube", lambda q, long_mix: {"id": "abcdefghijk", "title": q, "long": long_mix})
    h = pc.Hands(tmp_path / "m.md", tmp_path / "t.json", load_apps=False, spotify=_spotify(tmp_path, _FakeSpotify()))
    plan = h.find_music({"do": "play_music", "text": "soft romantic R&B", "number": 1})
    assert plan == {"spotify": {"name": "Romantic R&B", "context_uri": "spotify:playlist:rnb", "shuffle": True}}
    assert "youtube" in h.find_music({"do": "youtube", "text": "lofi", "number": 1})          # asked for YouTube
    assert h.find_music({"do": "play_music", "text": "party", "number": 4}) == \
        {"error": "I can't find party on your Spotify"}
    off = pc.Hands(tmp_path / "m.md", tmp_path / "t.json", load_apps=False)
    assert off.find_music({"do": "play_music", "text": "soft jazz", "number": 1})["youtube"]["long"] is True
    assert "youtube" in pc.ACTIONS and "spotify" not in pc.ACTIONS


def test_a_romantic_evening_opens_with_careless_whisper_then_the_romantic_playlist(tmp_path):
    fake = _FakeSpotify()
    real = fake.__call__
    playing = [{"is_playing": True, "progress_ms": 60_000, "item": {"uri": "spotify:track:1", "duration_ms": 300_000}},
               {"is_playing": True, "progress_ms": 297_500, "item": {"uri": "spotify:track:1", "duration_ms": 300_000}}]

    def http(method, url, body):
        if "/me/player/currently-playing" in url:
            return 200, playing.pop(0) if playing else {}
        return real(method, url, body)

    s = _spotify(tmp_path, http)
    s._sleep = lambda secs: None
    song, mood = s.find("Careless Whisper George Michael", 0), s.find("soft romantic R&B", 1)
    s.play(song)                                                     # the song first ...
    assert ("PUT", "/me/player/play?device_id=pc",
            {"context_uri": "spotify:album:A", "offset": {"uri": "spotify:track:1"}}) in fake.calls
    assert s._follow(song["offset_uri"], mood, "pc", 60) is True     # ... 2.5 s before it ends: the playlist
    assert ("PUT", "/me/player/play?device_id=pc", {"context_uri": "spotify:playlist:rnb"}) in fake.calls
    # the owner put on something else meanwhile: Vertex leaves it alone
    fake.calls.clear()
    playing[:] = [{"is_playing": True, "progress_ms": 1_000, "item": {"uri": "spotify:track:OTHER", "duration_ms": 9}}]
    assert s._follow(song["offset_uri"], mood, "pc", 60) is False
    assert not [c for c in fake.calls if c[0] == "PUT"]


def test_remembered_names_help_the_speech_model_spell_them():
    facts = ("The owner's girlfriend is named Safi. (2026-10-16)\nWhen the owner says \"romantic evening\": start "
             "with Careless Whisper by George Michael, then romantic music.")
    assert vv.memory_names(facts) == "Safi, Careless, Whisper, George, Michael"
    assert vv.memory_names("") == ""
