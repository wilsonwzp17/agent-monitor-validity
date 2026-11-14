"""Tests for the ended-response rule: a transport failure whose bytes hold the provider's
own end of a response is kept and halts until a correction note, in place of a retry.

Every response here is a synthetic, provider-shaped stream, response object or error body
served by FakeTransport or by the stage runner's fake provider. Nothing is sent.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage as RS  # noqa: E402
import test_dispatch as TD  # noqa: E402  LogCase, Fakes and the error bodies
import test_run_stage as TRS  # noqa: E402  RunnerCase, items and reserves
from judge import dispatch as D  # noqa: E402
from judge import requests as R  # noqa: E402
from judge.outcomes import FILTERED, HALT, TRANSPORT, VALID, Exchange, canonical_outcome  # noqa: E402
from judge.storage import AttemptLog  # noqa: E402
from judge.streams import anthropic_events, holds_end, openai_events, parse_sse, sse  # noqa: E402
from test_requests import answer  # noqa: E402

KEY, WS = TD.KEY, TD.WS
CANARY = "CANARY_ENDED_ANSWER_9d3"
TEXT = answer(verdict="failure", rationale=CANARY)
SERVER_ERROR = {"code": "server_error", "message": "The server had an error while processing your request."}
A_ERROR_EVENT = ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
B_ERROR_EVENT = ("error", {"type": "error", "code": "server_error", "message": "x", "param": None})
A_HEADERS = {"anthropic-workspace-id": WS}
PAUSED = "the bytes hold the provider's end of a response"


def a_stream(**kw):
    return anthropic_events(TEXT, msg_id="msg_ended", **kw)


def b_stream(text=TEXT, **kw):
    return openai_events(text, resp_id="resp_ended", **kw)


def a_object(stop_reason="end_turn"):
    """A whole Messages response as one JSON object, as a request without streaming gets it."""
    return json.dumps({"id": "msg_ended", "type": "message", "role": "assistant", "model": R.A_MODEL,
                       "content": [{"type": "text", "text": TEXT}], "stop_reason": stop_reason,
                       "stop_sequence": None, "usage": {"input_tokens": 1000, "output_tokens": 250}}).encode()


def b_object(text=TEXT, **kw):
    """A whole Responses object as one JSON object: the response a stream's terminal event carries."""
    return json.dumps(b_stream(text, **kw)[-1][1]["response"]).encode()


def bare_failed(error=SERVER_ERROR, output=()):
    """The events of a B stream that ends in ``response.failed`` with no usage and no answer."""
    events = b_stream(None, status="failed", error=error)
    events[-1][1]["response"].update(usage=None, output=list(output))
    return events


def ok(events, headers=None):
    return Exchange(200, headers or {}, sse(events), None, None)


def ended_attempts():
    """(name, judge, exchange) for attempts that the outcome table calls TRANSPORT and whose
    bytes hold the provider's end: shapes 1 to 4 of the boundary check."""
    done_a, done_b = a_stream(), b_stream()
    cut_short = openai_events(None, resp_id="resp_ended", status="incomplete", incomplete_reason="max_output_tokens")
    refused = openai_events(None, resp_id="resp_ended", status="failed", refusal="I can't help.", error=SERVER_ERROR)
    return [
        ("B failed with a transient code, carrying its answer", "B", ok(b_stream(status="failed", error=SERVER_ERROR))),
        ("B failed with no error object, carrying its answer", "B", ok(b_stream(status="failed", error=None))),
        ("B failed with a code in no list, carrying its answer", "B",
         ok(b_stream(status="failed", error={"code": "brand_new_code_2027", "message": "x"}))),
        ("B failed, carrying a refusal", "B", ok(refused)),
        ("B completed, then an error event", "B", ok(done_b + [B_ERROR_EVENT])),
        ("B an error event before the completed event", "B", ok(done_b[:-1] + [B_ERROR_EVENT] + done_b[-1:])),
        ("B incomplete, then an error event", "B", ok(cut_short + [B_ERROR_EVENT])),
        ("B completed, then failed", "B", ok(done_b + bare_failed()[-1:])),
        ("A a whole message, then an error event", "A", ok(done_a + [A_ERROR_EVENT], A_HEADERS)),
        ("A an error event before message_stop", "A", ok(done_a[:-1] + [A_ERROR_EVENT] + done_a[-1:], A_HEADERS)),
        ("B a 200 whose body is one response object", "B",
         Exchange(200, {"content-type": "application/json"}, b_object(), None, None)),
        ("A a 200 whose body is one message object", "A",
         Exchange(200, dict(A_HEADERS, **{"content-type": "application/json"}), a_object(), None, None)),
        ("B a 500 whose body is a whole stream", "B", Exchange(500, {}, sse(done_b), None, None)),
        ("A a 529 whose body is a whole stream", "A", Exchange(529, A_HEADERS, sse(done_a), None, None)),
    ]


def unended_attempts():
    """(name, judge, exchange) for transport failures with no provider end in their bytes."""
    raw_a, raw_b = sse(a_stream()), sse(b_stream())
    reset = "ConnectionResetError: reset"
    broken = sse(b_stream()[:-1]) + b'event: response.completed\ndata: {"type": "response.completed", "response": {\n\n'
    return [
        ("A no response", "A", TD.NO_RESPONSE),
        ("B no response", "B", TD.NO_RESPONSE),
        ("A a 200 with no bytes", "A", Exchange(200, A_HEADERS, b"", None, None)),
        ("A a 529 error body", "A", TD.a_err(529, "overloaded_error")),
        ("B a 500 error body", "B", TD.b_err(500, None, etype="server_error")),
        ("B a 429 rate limit body", "B", TD.b_err(429, "rate_limit_exceeded", etype="requests")),
        ("A a 200 whose body is an error object", "A", Exchange(200, A_HEADERS, TD.a_err(529, "overloaded_error").raw,
                                                                None, None)),
        ("B failed with no output and no usage", "B", ok(bare_failed())),
        ("B failed with no error object and no output", "B", ok(bare_failed(None))),
        ("B failed with a reasoning item only", "B", ok(b_stream(None, status="failed", error=SERVER_ERROR))),
        ("B a 200 whose body is a failed response object with no answer", "B",
         Exchange(200, {}, b_object(None, status="failed", error=SERVER_ERROR), None, None)),
        ("B a 200 whose body is a response object still in progress", "B",
         Exchange(200, {}, json.dumps(b_stream()[0][1]["response"]).encode(), None, None)),
        ("A cut after the stop reason, before message_stop", "A",
         Exchange(200, A_HEADERS, raw_a[:raw_a.index(b"event: message_stop")], reset, None)),
        ("A cut in the middle", "A", Exchange(200, A_HEADERS, raw_a[:len(raw_a) // 2], reset, None)),
        ("A message_stop with no stop reason", "A",
         Exchange(200, A_HEADERS, raw_a.replace(b'"stop_reason": "end_turn"', b'"stop_reason": null'), None, None)),
        ("B cut after the text, before the terminal event", "B",
         Exchange(200, {}, raw_b[:raw_b.index(b"event: response.completed")], reset, None)),
        ("B a terminal event that is not valid JSON", "B", Exchange(200, {}, broken, None, None)),
        ("A an error event and no end", "A", ok(anthropic_events(None)[:2] + [A_ERROR_EVENT], A_HEADERS)),
        ("B an error event and nothing else", "B", ok([B_ERROR_EVENT])),
    ]


def provider_of(judge):
    return "anthropic" if judge == "A" else "openai"


class TestHoldsEnd(unittest.TestCase):
    def test_the_shapes_the_dispatcher_pauses_hold_an_end_and_the_ones_it_retries_do_not(self):
        for name, judge, exchange in ended_attempts():
            self.assertTrue(holds_end(provider_of(judge), exchange.raw), name)
        for name, judge, exchange in unended_attempts():
            self.assertFalse(holds_end(provider_of(judge), exchange.raw), name)

    def test_an_ordinary_whole_stream_holds_its_end_wherever_it_opens(self):
        done_a, done_b = a_stream(), b_stream()
        for raw in (sse(done_a), sse([done_a[1], done_a[0]] + done_a[2:]), sse(done_a[1:])):
            self.assertTrue(holds_end("anthropic", raw))
        for raw in (sse(done_b), sse(done_b[2:]), sse(openai_events(None, status="incomplete",
                                                                    incomplete_reason="max_output_tokens"))):
            self.assertTrue(holds_end("openai", raw))

    def test_a_failed_response_ends_only_when_it_carries_an_answer(self):
        refusal_item = [{"type": "refusal", "refusal": "I can't help."}]
        empty_message = [{"id": "msg_1", "type": "message", "role": "assistant", "status": "in_progress",
                          "content": []}]
        cases = [(b_stream(status="failed", error=SERVER_ERROR), True),
                 (bare_failed(output=refusal_item), True),
                 (bare_failed(), False),
                 (bare_failed(output=[{"id": "rs_1", "type": "reasoning", "summary": []}]), False),
                 (bare_failed(output=empty_message), False)]
        for events, expected in cases:
            final = events[-1][1]["response"]
            self.assertEqual(holds_end("openai", sse(events)), expected, final["output"])
            self.assertEqual(holds_end("openai", json.dumps(final).encode()), expected, final["output"])

    def test_one_whole_response_object_holds_its_end_by_its_stop_reason_or_status(self):
        self.assertTrue(holds_end("anthropic", a_object()))
        self.assertTrue(holds_end("anthropic", a_object("max_tokens")))
        self.assertFalse(holds_end("anthropic", a_object(None)))
        self.assertTrue(holds_end("openai", b_object()))
        self.assertTrue(holds_end("openai", b_object(None, status="incomplete", incomplete_reason="max_output_tokens")))
        for status in ("in_progress", "queued", "cancelled", None):
            changed = dict(json.loads(b_object()), status=status)
            self.assertFalse(holds_end("openai", json.dumps(changed).encode()), status)
        self.assertFalse(holds_end("openai", json.dumps(dict(json.loads(b_object()), object="list")).encode()))
        self.assertFalse(holds_end("anthropic", json.dumps(dict(json.loads(a_object()), type="error")).encode()))

    def test_each_provider_s_end_is_read_only_as_its_own(self):
        self.assertFalse(holds_end("anthropic", sse(b_stream())))
        self.assertFalse(holds_end("anthropic", b_object()))
        self.assertFalse(holds_end("openai", sse(a_stream())))
        self.assertFalse(holds_end("openai", a_object()))

    def test_bytes_that_are_no_response_hold_no_end_and_do_not_raise(self):
        odd_a = [("message_delta", {"type": "message_delta", "delta": "end_turn"}),
                 ("message_stop", {"type": "message_stop"})]
        odd_b = [[("response.failed", {"type": "response.failed", "response": response})]
                 for response in ("failed", None, {"output": 5}, {"output": "message"},
                                  {"output": ["message", 5, None]}, {"output": {"type": "message", "content": "x"}})]
        for provider in ("anthropic", "openai"):
            for raw in (b"", b"not json", b"[]", b"null", b"5", b'"message"', b"\xff\xfe", b"[" * 200000,
                        b"event: x\ndata: [1]\n\n", sse(odd_a), *map(sse, odd_b)):
                self.assertFalse(holds_end(provider, raw), (provider, raw[:40]))


class DispatchCase(TD.LogCase):
    """The dispatcher's LogCase, with one slot's first attempt scripted and a fresh log per case."""

    def fresh(self, n):
        self.log = AttemptLog(self.tmp / f"log-{n}")
        self.sleeps.clear()

    def lead(self, judge):
        return [] if judge == "A" else [self.f.a(), self.f.a(), self.f.a()]

    def rest(self, judge):
        return [self.f.a(), self.f.a(), self.f.a(), self.f.b()] if judge == "A" else [self.f.b()]

    def outcomes(self, slot):
        return [r["outcome"] for r in self.records(slot)]


class TestTheDispatcherPausesAnEndedResponse(DispatchCase):
    def test_the_outcome_table_still_calls_each_of_these_a_transport_failure(self):
        for name, judge, exchange in ended_attempts() + unended_attempts():
            oc = canonical_outcome(provider_of(judge), exchange)
            self.assertEqual((oc["outcome"], oc["retryable"], oc["parse"]), (TRANSPORT, True, None), name)

    def test_each_is_sent_once_kept_and_sent_again_only_after_a_note(self):
        for n, (name, judge, first) in enumerate(ended_attempts()):
            self.fresh(n)
            slot, spare = f"{judge}1", (self.f.a() if judge == "A" else self.f.b())
            lead = self.lead(judge)
            result, t = self.run_slots(*lead, first, spare)
            self.assertTrue(result["halted"], name)
            self.assertEqual([r.judge for r in t.sent], ["A"] * len(lead) + [judge], name)     # one send of the slot
            self.assertEqual((t.script, self.sleeps), ([spare], []), name)                    # no retry, no backoff
            rec, = self.records(slot)
            self.assertEqual((rec["outcome"], rec["response_outcome"], rec["halt_kind"], rec["parse"],
                              rec["failed_checks"]), (HALT, HALT, "ended_response", None, []), name)
            self.assertFalse(D._slot_done([rec]), name)
            self.assertNotIn(CANARY, json.dumps(rec), name)                # nothing of the answer is in the record
            folder = self.log.slot_dir(KEY, self.config, judge, 1)
            self.assertEqual(sorted(p.name for p in folder.iterdir()), ["1.json", "1.raw"], name)
            self.assertEqual((folder / "1.raw").read_bytes(), first.raw, name)
            self.assertEqual(rec["raw_sha256"], R.sha256_bytes(first.raw), name)
            halt, = self.log.halts()
            self.assertEqual((halt["halt_id"], halt["slot"], halt["attempt"], halt["halt_kind"], halt["http_status"]),
                             (result["halt_id"], slot, 1, "ended_response", first.status), name)
            self.assertEqual(halt["detail"], rec["detail"], name)
            self.assertTrue(halt["detail"].endswith("; " + PAUSED), name)
            self.assertIn("next attempt index", halt["resume_rule"], name)
            for key in (KEY, "t1111111111111111"):                         # no trajectory starts without a note
                again = D.FakeTransport([spare])
                with self.assertRaises(D.HaltPending):
                    D.dispatch_trajectory(key, self.reqs, self.config, again, self.log, WS, sleep=self.sleeps.append,
                                          clock=self.clock)
                self.assertEqual(again.sent, [], name)
            self.log.record_correction(result["halt_id"], "Read 1.raw.", self.config, at="now")
            result, t = self.run_slots(*self.rest(judge))
            self.assertFalse(result["halted"], name)
            self.assertEqual([r.judge for r in t.sent], ["A", "A", "A", "B"][3 * (judge == "B"):], name)
            self.assertEqual((self.outcomes(slot), self.sleeps), ([HALT, VALID], []), name)
            self.assertEqual(self.records(slot)[1]["attempt"], 2, name)
            self.assertEqual((folder / "1.raw").read_bytes(), first.raw, name)     # the paused response is still there

    def test_the_record_keeps_what_the_stream_reported_and_the_table_s_reason(self):
        failed = ok(b_stream(status="failed", error=SERVER_ERROR))
        result, _ = self.run_slots(*self.lead("B"), failed)
        rec, = self.records("B1")
        self.assertEqual((rec["detail"], rec["stop"], rec["response_id"], rec["response_model"], rec["http_status"]),
                         ("response failed: server_error; " + PAUSED, "response.failed", "resp_ended", R.B_MODEL, 200))
        self.assertEqual((rec["error"]["code"], rec["usage"]["input_tokens"], rec["usage"]["output_tokens"]),
                         ("server_error", 1000, 250))
        raw = (self.log.slot_dir(KEY, self.config, "B", 1) / "1.raw").read_bytes()
        self.assertEqual((raw.count(CANARY.encode()), CANARY in json.dumps(rec)), (2, False))
        # No check ran on it, so its id is not claimed and it does not fix the first model string.
        self.assertFalse(self.log._response_id_path("openai", "resp_ended").exists())
        self.assertIsNone(self.log.first_model(self.config, "B"))

    def test_a_pause_uses_no_retry(self):
        ended = ok(b_stream(status="failed", error=SERVER_ERROR))
        result, _ = self.run_slots(*self.lead("B"), TD.NO_RESPONSE, ended)
        self.assertEqual((result["halted"], self.outcomes("B1"), self.sleeps), (True, [TRANSPORT, HALT], [2.0]))
        self.log.record_correction(result["halt_id"], "Read 2.raw.", self.config, at="now")
        result, t = self.run_slots(TD.NO_RESPONSE, TD.NO_RESPONSE, TD.NO_RESPONSE)
        self.assertFalse(result["halted"])
        self.assertEqual(self.outcomes("B1"), [TRANSPORT, HALT, TRANSPORT, TRANSPORT, TRANSPORT])
        self.assertEqual((result["slots"]["B1"]["outcome"], self.sleeps), (TRANSPORT, [2.0, 4.0, 8.0]))

    def test_the_helper_changes_only_a_transport_failure_whose_bytes_hold_an_end(self):
        ended, bare = ok(b_stream(status="failed", error=SERVER_ERROR)), ok(bare_failed())
        oc = canonical_outcome("openai", ended)
        paused = D._pause_ended("openai", ended.raw, oc)
        self.assertEqual((paused["outcome"], paused["retryable"], paused["halt_kind"]), (HALT, False, "ended_response"))
        self.assertEqual({k: paused[k] for k in ("summary", "parse", "error")},
                         {k: oc[k] for k in ("summary", "parse", "error")})
        self.assertEqual((oc["outcome"], oc["retryable"], oc["halt_kind"]), (TRANSPORT, True, None))   # left as it was
        same = canonical_outcome("openai", bare)
        self.assertIs(D._pause_ended("openai", bare.raw, same), same)
        valid = canonical_outcome("openai", ok(b_stream()))
        self.assertIs(D._pause_ended("openai", sse(b_stream()), valid), valid)


class TestTheDispatcherStillRetries(DispatchCase):
    def test_a_transport_failure_with_no_provider_end_is_retried_as_before(self):
        for n, (name, judge, first) in enumerate(unended_attempts()):
            self.fresh(n)
            lead = self.lead(judge)
            result, t = self.run_slots(*lead, first, *self.rest(judge))
            self.assertFalse(result["halted"], name)
            self.assertEqual([r.judge for r in t.sent],
                             ["A"] * len(lead) + [judge] + ["A", "A", "A", "B"][3 * (judge == "B"):], name)
            self.assertEqual(t.sent[len(lead)].body, t.sent[len(lead) + 1].body, name)          # the same bytes again
            self.assertEqual((self.outcomes(f"{judge}1"), self.sleeps, self.log.halts()),
                             ([TRANSPORT, VALID], [2.0], []), name)
            self.assertIsNone(self.records(f"{judge}1")[0]["halt_kind"], name)

    def test_four_bare_failed_responses_in_a_row_still_end_the_slot_as_invalid(self):
        result, t = self.run_slots(*self.lead("B"), *[ok(bare_failed()) for _ in range(4)])
        self.assertFalse(result["halted"])
        self.assertEqual((self.outcomes("B1"), self.sleeps, self.log.halts()), ([TRANSPORT] * 4, [2.0, 4.0, 8.0], []))
        self.assertEqual(result["slots"]["B1"]["transport_attempts"], 4)

    def test_an_attempt_with_no_bytes_is_retried_without_being_read(self):
        with mock.patch.object(D, "holds_end", side_effect=AssertionError("there are no bytes to read")) as reader:
            result, _ = self.run_slots(TD.NO_RESPONSE, Exchange(None, {}, None, "TimeoutError: timed out", None),
                                       self.f.a(), self.f.a(), self.f.a(), self.f.b())
            self.log.write_raw("t2222222222222222", self.config, "A", 1, 1, b"")
            rec = D.recover_attempt(self.log, reservation("t2222222222222222", "A", self.reqs))
        self.assertFalse(result["halted"])
        self.assertEqual((self.outcomes("A1"), rec["outcome"], rec["halt_kind"]),
                         ([TRANSPORT, TRANSPORT, VALID], TRANSPORT, None))
        reader.assert_not_called()

    def test_what_the_table_already_decides_keeps_its_outcome_and_halt_kind(self):
        def failed(code):
            return ok(b_stream(status="failed", error={"code": code, "message": "x"}))

        raw_b = sse(b_stream())
        cases = [("a whole stream, then the connection reset",
                  Exchange(200, {}, raw_b, "ConnectionResetError: reset", None), False, VALID, None),
                 ("failed with a code that halts, carrying its answer", failed("invalid_image"), True, HALT,
                  "request_error"),
                 ("failed with a policy code, carrying its answer", failed("invalid_prompt"), False, FILTERED, None),
                 ("a 400 whose body is a whole stream", Exchange(400, {}, raw_b, None, None), True, HALT,
                  "request_error"),
                 ("a whole stream on another tier", ok(b_stream(service_tier="flex")), True, HALT, "response_check")]
        for n, (name, first, halted, outcome, kind) in enumerate(cases):
            self.fresh(n)
            result, t = self.run_slots(*self.lead("B"), first, self.f.b())
            self.assertEqual((result["halted"], [r.judge for r in t.sent]), (halted, ["A", "A", "A", "B"]), name)
            rec, = self.records("B1")
            self.assertEqual((rec["outcome"], rec["halt_kind"]), (outcome, kind), name)


def reservation(key, judge, reqs):
    """What ``recover_attempt`` takes from the ledger for a slot's first attempt."""
    return {"trace_key": key, "config_sha256": R.config_sha256(), "judge": judge, "repeat": 1, "attempt": 1,
            "provider": provider_of(judge), "model": R.A_MODEL if judge == "A" else R.B_MODEL,
            "request_sha256": reqs[judge].sha256}


class TestRecovery(DispatchCase):
    """A stop between the raw bytes and the record: ``recover_attempt`` rebuilds the record."""

    def rebuilt(self, n, judge, raw):
        self.fresh(n)
        self.log.write_raw(KEY, self.config, judge, 1, 1, raw)
        returned = D.recover_attempt(self.log, reservation(KEY, judge, self.reqs))
        rec, = self.records(f"{judge}1")
        self.assertEqual((rec["outcome"], rec["halt_kind"], rec["detail"]),
                         (returned["outcome"], returned["halt_kind"], returned["detail"]))
        self.assertEqual((self.log.slot_dir(KEY, self.config, judge, 1) / "1.raw").read_bytes(), raw)
        self.assertEqual(rec["raw_sha256"], R.sha256_bytes(raw))
        return rec

    def test_rebuilt_bytes_that_hold_an_end_are_a_halt_and_their_slot_stays_open(self):
        done_a = a_stream()
        cases = [("A a whole stream that opens with a ping", "A", sse([done_a[1], done_a[0]] + done_a[2:]), None),
                 ("A one message object", "A", a_object(), None),
                 ("B one response object", "B", b_object(), None),
                 ("B a whole stream that lost its opening events", "B", sse(b_stream()[2:]), None),
                 ("B failed, carrying its answer", "B", sse(b_stream(status="failed", error=SERVER_ERROR)), 200),
                 ("B completed, then an error event", "B", sse(b_stream() + [B_ERROR_EVENT]), 200),
                 ("A a whole message, then an error event", "A", sse(done_a + [A_ERROR_EVENT]), 200)]
        for n, (name, judge, raw, status) in enumerate(cases):
            rec = self.rebuilt(n, judge, raw)
            self.assertEqual((rec["outcome"], rec["halt_kind"], rec["http_status"], rec["parse"], rec["failed_checks"]),
                             (HALT, "ended_response", status, None, []), name)
            self.assertEqual(rec["recovered"]["http_status"],
                             "inferred from the stream" if status else "unknown; taken as no response", name)
            self.assertTrue(rec["detail"].endswith("; " + PAUSED), name)
            self.assertNotIn(CANARY, json.dumps(rec), name)
            self.assertFalse(D._slot_done([rec]), name)
            self.assertIn("next attempt index", D.resume_rule(rec["outcome"]), name)

    def test_the_stream_the_live_dispatcher_calls_valid_is_the_one_a_rebuild_cannot_recognize(self):
        done = a_stream()
        raw = sse([done[1], done[0]] + done[2:])                       # the ping comes before message_start
        self.assertEqual(canonical_outcome("anthropic", Exchange(200, {}, raw, None, None))["outcome"], VALID)
        self.assertFalse(D._stream_started("anthropic", raw))
        self.assertEqual(self.rebuilt(0, "A", raw)["detail"],
                         "no response: the response's status was not kept; " + PAUSED)

    def test_rebuilt_bytes_with_no_provider_end_are_still_a_transport_failure(self):
        raw_a = sse(a_stream())
        cases = [("A cut after the stop reason", "A", raw_a[:raw_a.index(b"event: message_stop")], 200),
                 ("A cut in the middle", "A", raw_a[:len(raw_a) // 2], 200),
                 ("B failed with no answer", "B", sse(bare_failed()), 200),
                 ("B an error event and nothing else", "B", sse([B_ERROR_EVENT]), 200),
                 ("B a transient error body", "B", TD.b_err(429, "rate_limit_exceeded", etype="requests").raw, None),
                 ("A an overloaded error body", "A", TD.a_err(529, "overloaded_error").raw, None),
                 ("B bytes that hold no response", "B", b"bytes that hold no response", None)]
        for n, (name, judge, raw, status) in enumerate(cases):
            rec = self.rebuilt(n, judge, raw)
            self.assertEqual((rec["outcome"], rec["halt_kind"], rec["http_status"]), (TRANSPORT, None, status), name)

    def test_a_rebuilt_whole_stream_that_opens_as_listed_is_recorded_as_it_was(self):
        rec = self.rebuilt(0, "B", sse(b_stream()))
        self.assertEqual((rec["outcome"], rec["halt_kind"], rec["parse"]["verdict"]), (VALID, None, "failure"))
        self.assertTrue(D._slot_done([rec]))


class OneSend(RS.ScriptedProvider):
    """The stage runner's fake provider, with the first attempt of one slot changed by ``change``."""

    def __init__(self, slot, change):
        super().__init__("central")
        self.slot, self.change = slot, change

    def _exchange(self, request, stage, ref, kind):
        exchange = super()._exchange(request, stage, ref, kind)
        return self.change(exchange) if ref.endswith(f"_{self.slot}_1") else exchange


def failed_with_its_answer(exchange):
    """A completed B exchange whose terminal event is turned into ``response.failed`` with a
    transient code. The failed response still carries the whole answer and its usage."""
    events = [(name, json.loads(data)) for name, data in parse_sse(exchange.raw)]
    final = events[-1][1]
    final["type"] = "response.failed"
    final["response"].update(status="failed", error=SERVER_ERROR)
    return exchange._replace(raw=sse(events[:-1] + [("response.failed", final)]))


def ping_first(exchange):
    """A whole A exchange whose ``ping`` comes before its ``message_start``."""
    blocks = exchange.raw.split(b"\n\n")
    return exchange._replace(raw=b"\n\n".join([blocks[1], blocks[0]] + blocks[2:]))


def stop_before_the_record(slot):
    """Leaves ``slot``'s raw response on disk and stops before its attempt record is written."""
    real = AttemptLog.complete_attempt

    def write(log, record):
        if record["slot"] == slot:
            raise KeyboardInterrupt
        return real(log, record)
    return mock.patch.object(AttemptLog, "complete_attempt", write)


class TestTheStageRunner(TRS.RunnerCase):
    """``run_stage`` in fake mode on one synthetic item."""

    def setUp(self):
        super().setUp()
        self.led, self.its = self.ledger(), TRS.items(1, "a")
        self.key, self.config = self.its[0].trace_key, R.config_sha256()
        self.log = AttemptLog(self.root / "P1" / "attempt_log")

    def refused_and_nothing_sent(self):
        again = RS.ScriptedProvider()
        with self.assertRaisesRegex(RS.StageRefused, "halts without a correction note"):
            self.stage("P1", self.its, self.led, provider=again)
        self.assertEqual(again.sent, [])

    def reservations(self, slot):
        return [(e["attempt"], self.led.reservations[e["id"]]["state"]) for e in TRS.reserves(self.led)
                if e["slot"] == slot]

    def test_a_failed_b_response_that_carries_its_answer_stops_the_stage_until_a_note(self):
        report, provider = self.stage("P1", self.its, self.led, provider=OneSend("B1", failed_with_its_answer))
        self.assertEqual((report["complete"], report["stopped"]["kind"], report["stopped"]["slot"]),
                         (False, "ended_response", "B1"))
        self.assertEqual([r.judge for r in provider.sent], ["A", "A", "A", "B"])
        self.assertFalse((self.root / "P1" / "report.json").exists())
        rec, = self.log.attempts(self.key, self.config, "B", 1)
        self.assertEqual((rec["outcome"], rec["halt_kind"], rec["parse"], rec["stop"]),
                         (HALT, "ended_response", None, "response.failed"))
        raw = self.log.slot_dir(self.key, self.config, "B", 1) / "1.raw"
        kept = raw.read_bytes()
        self.assertEqual(R.sha256_bytes(kept), rec["raw_sha256"])
        self.assertIn(b"A scripted answer from the fake provider.", kept)
        self.refused_and_nothing_sent()
        self.log.record_correction(report["stopped"]["halt_id"], "Read 1.raw.", self.config, at="now")
        report, after = self.stage("P1", self.its, self.led)
        self.assertTrue(report["complete"], report["stopped"])
        self.assertEqual([r.judge for r in after.sent], ["B"])
        self.assertEqual([(r["attempt"], r["outcome"]) for r in self.log.attempts(self.key, self.config, "B", 1)],
                         [(1, HALT), (2, VALID)])
        self.assertEqual(raw.read_bytes(), kept)
        self.assertEqual([(h["slot"], h["kind"], h["correction"]) for h in report["gates"]["halts"]],
                         [("B1", "ended_response", "Read 1.raw.")])
        # The paused response reported its usage, so its call is settled, and the slot is paid for twice.
        self.assertEqual(self.reservations("B1"), [(1, "settled"), (2, "settled")])

    def test_a_bare_failed_b_response_is_still_retried_and_the_stage_completes(self):
        def bare(exchange):
            return exchange._replace(raw=sse(bare_failed()))

        report, provider = self.stage("P1", self.its, self.led, provider=OneSend("B1", bare))
        self.assertTrue(report["complete"], report["stopped"])
        self.assertEqual([r.judge for r in provider.sent], ["A", "A", "A", "B", "B"])
        self.assertEqual(provider.sent[3].body, provider.sent[4].body)
        self.assertEqual([(r["attempt"], r["outcome"]) for r in self.log.attempts(self.key, self.config, "B", 1)],
                         [(1, TRANSPORT), (2, VALID)])
        self.assertEqual(report["gates"]["halts"], [])

    def test_a_rebuilt_whole_stream_that_opens_with_a_ping_waits_for_a_note(self):
        with stop_before_the_record("A1"):
            with self.assertRaises(KeyboardInterrupt):
                self.stage("P1", self.its, self.led, provider=OneSend("A1", ping_first))
        raw = self.log.slot_dir(self.key, self.config, "A", 1) / "1.raw"
        kept = raw.read_bytes()
        self.assertEqual(canonical_outcome("anthropic", Exchange(200, {}, kept, None, None))["outcome"], VALID)
        self.assertEqual(self.log.attempts(self.key, self.config, "A", 1), [])
        self.refused_and_nothing_sent()
        rec, = self.log.attempts(self.key, self.config, "A", 1)
        self.assertEqual((rec["outcome"], rec["halt_kind"], rec["http_status"], rec["recovered"]["http_status"]),
                         (HALT, "ended_response", None, "unknown; taken as no response"))
        halt, = self.log.halts()
        self.assertEqual((halt["slot"], halt["attempt"], halt["halt_kind"], halt["recorded_at_start"]),
                         ("A1", 1, "ended_response", True))
        self.assertIn("next attempt index", halt["resume_rule"])
        self.refused_and_nothing_sent()
        self.log.record_correction(halt["halt_id"], "Read 1.raw.", self.config, at="now")
        report, after = self.stage("P1", self.its, self.led)
        self.assertTrue(report["complete"], report["stopped"])
        self.assertEqual([r.judge for r in after.sent], ["A", "A", "A", "B"])
        self.assertEqual([(r["attempt"], r["outcome"]) for r in self.log.attempts(self.key, self.config, "A", 1)],
                         [(1, HALT), (2, VALID)])
        self.assertEqual(raw.read_bytes(), kept)
        # Its status was not kept, so the paused call's reservation is held, as before the rule.
        self.assertEqual(self.reservations("A1"), [(1, "held"), (2, "settled")])


if __name__ == "__main__":
    unittest.main()
