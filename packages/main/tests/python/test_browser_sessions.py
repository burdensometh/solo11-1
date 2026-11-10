"""Tests for the isolated multi-window session management in
`RPA.Browser.sessions`.

Two layers are covered:

- deterministic unit tests with fake Selenium drivers (no browser needed),
  exercising state isolation, alert handling, cleanup and retries;
- real headless-Chrome integration tests, skipped when no browser/webdriver
  is available on the host.
"""

import re
import shutil
import threading

import pytest
from selenium.common.exceptions import (
    NoAlertPresentException,
    NoSuchElementException,
    NoSuchWindowException,
    TimeoutException,
    WebDriverException,
)

from RPA.Browser.Selenium import Selenium
from RPA.Browser.sessions import BrowserState, SessionKeywords

from . import RESOURCES_DIR


FIXTURE_PAGE = f"file://{RESOURCES_DIR / 'browser_sessions_fixture.html'}"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeAlert:
    def __init__(self, text="alert text", accept_error=None):
        self.text = text
        self._accept_error = accept_error
        self.accepted = False
        self.dismissed = False

    def accept(self):
        self.accepted = True
        if self._accept_error:
            raise self._accept_error

    def dismiss(self):
        self.dismissed = True


class FakeSwitchTo:
    def __init__(self, owner):
        self._owner = owner
        self._alert = owner.alert

    @property
    def alert(self):
        alert = self._owner.alert
        if alert is None:
            raise NoAlertPresentException("no alert")
        return alert

    def window(self, handle):
        if handle not in self._owner.handles:
            raise NoSuchWindowException(handle)
        self._owner.current_handle = handle

    def default_content(self):
        self._owner.default_content_called = True

    def frame(self, frame):
        self._owner.frames_entered.append(frame)


class FakeService:
    def __init__(self):
        self.process = FakeProcess()
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeProcess:
    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True


class FakeDriver:
    """Minimal Selenium driver stand-in with switch/window/alert support."""

    def __init__(self, handles=None, current=None, alert=None, session_id=None):
        self.handles = list(handles or ["w1"])
        self.current_handle = current or self.handles[0]
        self.alert = alert
        self.default_content_called = False
        self.frames_entered = []
        self.closed_handles = []
        self.quit_count = 0
        self.session_id = session_id or "session"
        self.service = FakeService()
        self.switch_to = FakeSwitchTo(self)
        self.scripts = []
        self.url = "https://example.test/start"
        self.ready_state = "complete"
        self.name = "chrome"
        # When set, every driver command raises (crashed/dead session).
        self.dead_error = None

    # window handles ----------------------------------------------------- #
    @property
    def window_handles(self):
        if self.dead_error:
            raise self.dead_error
        return list(self.handles)

    @property
    def current_window_handle(self):
        if self.dead_error:
            raise self.dead_error
        return self.current_handle

    @property
    def current_url(self):
        if self.dead_error:
            raise self.dead_error
        return self.url

    def close(self):
        if self.dead_error:
            raise self.dead_error
        self.closed_handles.append(self.current_handle)
        self.handles.remove(self.current_handle)

    def quit(self):
        if self.dead_error:
            raise self.dead_error
        self.quit_count += 1

    def execute_script(self, script, *args):
        if self.dead_error:
            raise self.dead_error
        self.scripts.append(script)
        if "readyState" in script:
            return self.ready_state
        return None


class FakeBrowserManagement:
    def __init__(self, timeout=5.0):
        self._timeout = timeout
        self.navigations = []

    def get_timeout(self, timeout=None):
        if timeout is None:
            return self._timeout
        if isinstance(timeout, (int, float)):
            return float(timeout)
        match = re.fullmatch(
            r"\s*([0-9]*\.?[0-9]+)\s*(ms|s|sec|secs)?\s*",
            str(timeout).lower(),
        )
        if not match:
            return float(timeout)
        value = float(match.group(1))
        return value / 1000 if match.group(2) == "ms" else value

    def go_to(self, url):
        self.navigations.append(url)


def make_library(drivers=None, current_index=None, timeout=5.0):
    """Build a Selenium instance wired with fake drivers in its cache."""
    library = Selenium.__new__(Selenium)
    # Only the attributes touched by SessionKeywords are initialised.
    library._drivers = FakeCache(drivers or [], current_index=current_index)
    library.browser_management = FakeBrowserManagement(timeout=timeout)
    library.go_to = library.browser_management.go_to
    sessions = SessionKeywords(library)
    library.sessions = sessions
    return library, sessions


class _NoCurrent:
    """Mimics robot.utils.ConnectionCache._no_current sentinel."""

    def __bool__(self):
        return False

    def __getattr__(self, name):
        raise RuntimeError("No current browser")


class FakeCache:
    """Subset of robot.utils.ConnectionCache / WebDriverCache used here."""

    def __init__(self, drivers, current_index=None):
        self._connections = list(drivers)
        self._closed = set()
        self._aliases = {}
        self._no_current = _NoCurrent()
        self.current_index = current_index if current_index is not None else (
            len(drivers) if drivers else None
        )
        self.current = (
            self._connections[self.current_index - 1]
            if self.current_index
            else self._no_current
        )

    def switch(self, identifier):
        if isinstance(identifier, str):
            identifier = self._aliases.get(identifier, identifier)
        index = int(identifier)
        try:
            driver = self._connections[index - 1]
        except IndexError:
            # Matches robot.utils.ConnectionCache.switch behaviour.
            raise RuntimeError(
                f"No browser with index or alias '{identifier}' found."
            ) from None
        self.current_index = index
        self.current = driver

    def get_index(self, alias_or_index):
        try:
            if isinstance(alias_or_index, str):
                if alias_or_index in self._aliases:
                    alias_or_index = self._aliases[alias_or_index]
                else:
                    return None
            index = int(alias_or_index)
            driver = self._connections[index - 1]
        except (ValueError, IndexError, KeyError):
            return None
        return None if driver in self._closed else index

    def get_connection(self, index):
        return self._connections[index - 1]

    def register(self, driver, alias=None):
        self._connections.append(driver)
        index = len(self._connections)
        if alias:
            self._aliases[alias] = index
        self.current_index = index
        return index

    def empty_cache(self):
        self._connections = []
        self._closed = set()
        self._aliases = {}
        self.current_index = None
        self.current = self._no_current


# --------------------------------------------------------------------------- #
# State snapshots
# --------------------------------------------------------------------------- #
class TestBrowserState:
    def test_save_and_restore_state(self):
        driver = FakeDriver(handles=["w1", "w2"], current="w1")
        _, sessions = make_library([driver], current_index=1)

        state = sessions.save_browser_state()
        assert isinstance(state, BrowserState)
        assert state.index == 1
        assert state.handle == "w1"

        driver.switch_to.window("w2")
        assert driver.current_window_handle == "w2"

        assert sessions.restore_browser_state(state) is True
        assert driver.current_window_handle == "w1"
        assert driver.default_content_called

    def test_restore_returns_false_when_browser_gone(self):
        driver = FakeDriver()
        library, sessions = make_library([driver], current_index=1)
        state = sessions.save_browser_state()
        library._drivers.empty_cache()

        assert sessions.restore_browser_state(state) is False

    def test_restore_returns_false_when_window_closed(self):
        driver = FakeDriver(handles=["w1", "w2"], current="w1")
        _, sessions = make_library([driver], current_index=1)
        state = sessions.save_browser_state()
        driver.handles.remove("w1")

        assert sessions.restore_browser_state(state) is False

    def test_browser_state_object_restores_directly(self):
        driver = FakeDriver(handles=["w1", "w2"], current="w2")
        state = BrowserState(index=1, handle="w1")
        state.restore(driver)
        assert driver.current_window_handle == "w1"
        assert driver.default_content_called


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
class TestAlerts:
    def test_dismiss_any_alert_accepts(self):
        alert = FakeAlert("hello")
        driver = FakeDriver(alert=alert)
        _, sessions = make_library([driver])

        assert sessions.dismiss_any_alert() == "hello"
        assert alert.accepted

    def test_dismiss_any_alert_dismisses(self):
        alert = FakeAlert()
        driver = FakeDriver(alert=alert)
        _, sessions = make_library([driver])

        sessions.dismiss_any_alert(action="DISMISS")
        assert alert.dismissed

    def test_dismiss_any_alert_leave_keeps_dialog(self):
        alert = FakeAlert()
        driver = FakeDriver(alert=alert)
        _, sessions = make_library([driver])

        sessions.dismiss_any_alert(action="LEAVE")
        assert not alert.accepted and not alert.dismissed

    def test_dismiss_any_alert_returns_none_when_absent(self):
        driver = FakeDriver(alert=None)
        _, sessions = make_library([driver])

        assert sessions.dismiss_any_alert(timeout=0) is None

    def test_dismiss_any_alert_waits_for_late_alert(self):
        driver = FakeDriver(alert=None)
        _, sessions = make_library([driver])
        alert = FakeAlert("late")

        def arm_alert():
            driver.alert = alert

        timer = threading.Timer(0.15, arm_alert)
        timer.start()
        try:
            assert sessions.dismiss_any_alert(timeout=2) == "late"
            assert alert.accepted
        finally:
            timer.cancel()

    def test_dismiss_any_alert_invalid_action(self):
        driver = FakeDriver(alert=FakeAlert())
        _, sessions = make_library([driver])
        with pytest.raises(ValueError):
            sessions.dismiss_any_alert(action="BOGUS")


# --------------------------------------------------------------------------- #
# Window discovery and safe closing
# --------------------------------------------------------------------------- #
class TestWindowSwitching:
    def test_switch_to_new_window_with_known_handles(self):
        driver = FakeDriver(handles=["w1"], current="w1")
        _, sessions = make_library([driver])

        def open_popup():
            driver.handles.append("w2")

        timer = threading.Timer(0.1, open_popup)
        timer.start()
        try:
            handle = sessions.switch_to_new_window(
                timeout=2, known_handles=["w1"]
            )
        finally:
            timer.join()
        assert handle == "w2"
        assert driver.current_window_handle == "w2"

    def test_switch_to_new_window_timeout(self):
        driver = FakeDriver(handles=["w1"], current="w1")
        _, sessions = make_library([driver])
        with pytest.raises(TimeoutException):
            sessions.switch_to_new_window(
                timeout=0.1, known_handles=["w1"]
            )

    def test_switch_to_new_window_no_switch_returns_handle(self):
        driver = FakeDriver(handles=["w1", "w2"], current="w1")
        _, sessions = make_library([driver])
        handle = sessions.switch_to_new_window(
            timeout=0.1, known_handles=["w1"], switch=False
        )
        assert handle == "w2"
        assert driver.current_window_handle == "w1"

    def test_switch_to_new_window_without_known_handles(self):
        driver = FakeDriver(handles=["w1", "w2"], current="w1")
        _, sessions = make_library([driver])
        assert sessions.switch_to_new_window(timeout=0.1) == "w2"

    def test_close_window_safely_last_window_quits_browser(self):
        driver = FakeDriver(handles=["w1"], current="w1")
        _, sessions = make_library([driver])
        assert sessions.close_window_safely() is True
        assert driver.quit_count == 1

    def test_close_window_safely_closes_and_switches_to_remaining(self):
        driver = FakeDriver(handles=["w1", "w2"], current="w2")
        _, sessions = make_library([driver])
        assert sessions.close_window_safely() is True
        assert driver.closed_handles == ["w2"]
        assert driver.current_window_handle == "w1"

    def test_close_window_safely_dismisses_alert_first(self):
        alert = FakeAlert()
        driver = FakeDriver(handles=["w1", "w2"], current="w2", alert=alert)
        _, sessions = make_library([driver])
        sessions.close_window_safely()
        assert alert.accepted
        assert driver.closed_handles == ["w2"]

    def test_close_window_safely_restores_given_state(self):
        driver = FakeDriver(handles=["w1", "w2"], current="w2")
        _, sessions = make_library([driver])
        state = BrowserState(index=1, handle="w1")
        sessions.close_window_safely(restore_state=state)
        assert driver.current_window_handle == "w1"

    def test_close_window_safely_swallows_driver_error(self):
        driver = FakeDriver(handles=["w1", "w2"], current="w2")
        driver.dead_error = WebDriverException("boom")
        _, sessions = make_library([driver])
        # Must not raise: cleanup paths have to be failure-proof.
        assert sessions.close_window_safely() is False

    def test_browser_window_context_restores_state_on_success(self):
        driver = FakeDriver(handles=["w1", "w2"], current="w1")
        _, sessions = make_library([driver])
        with sessions.browser_window("w2"):
            assert driver.current_window_handle == "w2"
        assert driver.current_window_handle == "w1"

    def test_browser_window_context_restores_state_on_exception(self):
        driver = FakeDriver(handles=["w1", "w2"], current="w1")
        _, sessions = make_library([driver])
        with pytest.raises(RuntimeError, match="boom"):
            with sessions.browser_window("w2"):
                raise RuntimeError("boom")
        assert driver.current_window_handle == "w1"

    def test_browser_window_requires_handle_or_locator(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])
        with pytest.raises(ValueError):
            with sessions.browser_window():
                pass

    def test_window_scope_helper(self):
        driver = FakeDriver(handles=["w1", "w2"], current="w1")
        _, sessions = make_library([driver])
        with sessions.window_scope("w2"):
            assert driver.current_window_handle == "w2"
        assert driver.current_window_handle == "w1"

    def test_nested_window_scopes_restore_in_order(self):
        driver = FakeDriver(handles=["w1", "w2", "w3"], current="w1")
        _, sessions = make_library([driver])
        with sessions.browser_window("w2"):
            assert driver.current_window_handle == "w2"
            with sessions.browser_window("w3"):
                assert driver.current_window_handle == "w3"
            assert driver.current_window_handle == "w2"
        assert driver.current_window_handle == "w1"

    def test_interleaved_scopes_never_leak_window_state(self):
        # Simulate two interleaved flows (e.g. threads/generators) sharing one
        # browser: each must always return to its own saved window.
        driver = FakeDriver(
            handles=["main", "popup-a", "popup-b"], current="main"
        )
        _, sessions = make_library([driver])
        state_a = sessions.save_browser_state()
        driver.switch_to.window("popup-a")
        state_b = sessions.save_browser_state()
        driver.switch_to.window("popup-b")

        # Flow A restores while flow B is still inside its popup.
        assert sessions.restore_browser_state(state_a) is True
        assert driver.current_window_handle == "main"

        # Flow B must still be able to return to popup-a exactly.
        assert sessions.restore_browser_state(state_b) is True
        assert driver.current_window_handle == "popup-a"


# --------------------------------------------------------------------------- #
# Browser-level scopes and lifecycle
# --------------------------------------------------------------------------- #
class TestBrowserScopes:
    def test_browser_scope_switches_and_restores(self):
        first = FakeDriver(session_id="a")
        second = FakeDriver(session_id="b")
        cache = FakeCache([first, second], current_index=1)
        library = Selenium.__new__(Selenium)
        library._drivers = cache
        library.browser_management = FakeBrowserManagement()
        sessions = SessionKeywords(library)

        assert cache.current is first
        with sessions.browser_scope(2):
            assert cache.current is second
        assert cache.current is first

    def test_browser_scope_restores_after_exception(self):
        first = FakeDriver(session_id="a")
        second = FakeDriver(session_id="b")
        cache = FakeCache([first, second], current_index=1)
        library = Selenium.__new__(Selenium)
        library._drivers = cache
        library.browser_management = FakeBrowserManagement()
        sessions = SessionKeywords(library)

        with pytest.raises(ValueError):
            with sessions.browser_scope(2):
                raise ValueError("nope")
        assert cache.current is first


class TestSafeClose:
    def test_close_browser_safely_current(self):
        driver = FakeDriver()
        cache_driver, sessions = make_library([driver])
        del cache_driver
        assert sessions.close_browser_safely() is True
        assert driver.quit_count == 1
        assert driver in sessions._drivers._closed
        assert sessions._drivers.current_index is None
        assert not sessions._drivers.current

    def test_close_browser_safely_by_index_restores_selection(self):
        first = FakeDriver(session_id="a")
        second = FakeDriver(session_id="b")
        cache = FakeCache([first, second], current_index=2)
        library = Selenium.__new__(Selenium)
        library._drivers = cache
        library.browser_management = FakeBrowserManagement()
        sessions = SessionKeywords(library)

        assert sessions.close_browser_safely(1) is True
        assert first.quit_count == 1
        # The previously selected second browser stays active afterwards.
        assert cache.current_index == 2
        assert cache.current is second

    def test_close_browser_safely_closing_current_by_index_resets_current(self):
        first = FakeDriver(session_id="a")
        second = FakeDriver(session_id="b")
        cache = FakeCache([first, second], current_index=1)
        library = Selenium.__new__(Selenium)
        library._drivers = cache
        library.browser_management = FakeBrowserManagement()
        sessions = SessionKeywords(library)

        assert sessions.close_browser_safely(1) is True
        assert cache.current_index is None
        assert not cache.current

    def test_close_browser_safely_unknown_alias(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])
        assert sessions.close_browser_safely("ghost") is False
        assert driver.quit_count == 0

    def test_close_browser_safely_without_browser(self):
        _, sessions = make_library([])
        assert sessions.close_browser_safely() is False

    def test_close_browser_safely_force_stops_dead_driver(self):
        driver = FakeDriver()
        driver.dead_error = WebDriverException("session deleted")
        _, sessions = make_library([driver])
        assert sessions.close_browser_safely() is True
        assert driver.service.process.killed
        assert driver in sessions._drivers._closed

    def test_close_all_browsers_safely_quits_every_driver(self):
        first = FakeDriver(session_id="a")
        second = FakeDriver(session_id="b")
        second.dead_error = WebDriverException("dead")
        _, sessions = make_library([first, second])

        closed = sessions.close_all_browsers_safely()
        assert closed == 2
        assert first.quit_count == 1
        assert second.service.process.killed
        assert sessions._drivers._connections == []

    def test_close_all_browsers_safely_skips_already_closed(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])
        sessions._drivers._closed.add(driver)
        assert sessions.close_all_browsers_safely() == 0
        assert driver.quit_count == 0

    def test_prune_dead_browsers_keeps_live_ones(self):
        alive = FakeDriver(session_id="alive")
        dead = FakeDriver(session_id="dead")
        dead.dead_error = WebDriverException("crashed")
        _, sessions = make_library([alive, dead])

        pruned = sessions.prune_dead_browsers()
        assert pruned == [2]
        assert dead.service.process.killed
        assert dead in sessions._drivers._closed
        assert alive not in sessions._drivers._closed

    def test_prune_dead_browsers_all_healthy(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])
        assert sessions.prune_dead_browsers() == []


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #
class TestRetry:
    def test_run_with_retry_succeeds_on_second_attempt(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise WebDriverException("transient")
            return "ok"

        result = sessions.run_with_retry(flaky, retries=3, backoff=0.01)
        assert result == "ok"
        assert calls["n"] == 2

    def test_run_with_retry_runs_cleanup_between_attempts(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])
        cleanup_calls = []

        def action():
            if len(cleanup_calls) < 2:
                raise WebDriverException("transient")
            return "done"

        result = sessions.run_with_retry(
            action,
            retries=4,
            backoff=0.01,
            cleanup=lambda: cleanup_calls.append(1),
        )
        assert result == "done"
        # Cleanup runs before retry 2 and retry 3, not before attempt 1.
        assert len(cleanup_calls) == 2

    def test_run_with_retry_does_not_retry_selector_errors(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])
        calls = {"n": 0}

        def bad_locator():
            calls["n"] += 1
            raise NoSuchElementException("not found")

        with pytest.raises(NoSuchElementException):
            sessions.run_with_retry(bad_locator, retries=5, backoff=0.01)
        assert calls["n"] == 1

    def test_run_with_retry_exhausts_and_reraises(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])
        calls = {"n": 0}

        def always_fails():
            calls["n"] += 1
            raise WebDriverException("permanent")

        with pytest.raises(WebDriverException, match="permanent"):
            sessions.run_with_retry(always_fails, retries=3, backoff=0.01)
        assert calls["n"] == 3

    def test_run_with_retry_cleanup_exception_is_swallowed(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])
        calls = {"n": 0}

        def action():
            calls["n"] += 1
            if calls["n"] < 2:
                raise WebDriverException("transient")
            return True

        def broken_cleanup():
            raise RuntimeError("cleanup blew up")

        assert sessions.run_with_retry(
            action, retries=3, backoff=0.01, cleanup=broken_cleanup
        )

    def test_run_with_retry_custom_predicate(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])
        calls = {"n": 0}

        def action():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("retry me")
            return "fine"

        # ValueError is non-retryable by default; opt in explicitly.
        assert sessions.run_with_retry(
            action,
            retries=3,
            backoff=0.01,
            retry_on=ValueError,
        ) == "fine"
        assert calls["n"] == 3

    def test_run_with_retry_forwards_arguments(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])

        def add(a, b, c=0):
            return a + b + c

        assert sessions.run_with_retry(add, 1, 2, c=3, backoff=0.01) == 6

    def test_run_with_retry_rejects_non_callable(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])
        with pytest.raises(TypeError):
            sessions.run_with_retry(1234)

    def test_run_with_retry_accepts_robot_time_strings(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])
        # Numeric durations pass through; None falls back to the default.
        assert sessions._seconds(2, 1.0) == 2.0
        assert sessions._seconds(None, 1.0) == 1.0
        # Plain numeric and Robot-style time strings are both accepted.
        assert sessions._seconds("0.01", 1.0) == 0.01
        assert sessions._seconds("0.1s", 1.0) == 0.1
        assert sessions._seconds("250ms", 1.0) == 0.25
        result = sessions.run_with_retry(
            lambda: "ok", retries="3", backoff="0.01s"
        )
        assert result == "ok"


# --------------------------------------------------------------------------- #
# Page readiness / hardened navigation
# --------------------------------------------------------------------------- #
class TestPageReadiness:
    def test_wait_for_page_loaded_complete(self):
        driver = FakeDriver()
        driver.ready_state = "complete"
        _, sessions = make_library([driver])
        assert sessions.wait_for_page_loaded(timeout=1) == "complete"

    def test_wait_for_page_loaded_polls_until_ready(self):
        driver = FakeDriver()
        driver.ready_state = "loading"
        _, sessions = make_library([driver])

        def become_ready():
            driver.ready_state = "complete"

        timer = threading.Timer(0.15, become_ready)
        timer.start()
        try:
            assert sessions.wait_for_page_loaded(
                state="complete", timeout=2, poll_interval=0.05
            ) == "complete"
        finally:
            timer.join()

    def test_wait_for_page_loaded_timeout(self):
        driver = FakeDriver()
        driver.ready_state = "loading"
        _, sessions = make_library([driver])
        with pytest.raises(TimeoutException):
            sessions.wait_for_page_loaded(
                state="complete", timeout=0.1, poll_interval=0.05
            )

    def test_wait_for_page_loaded_invalid_state(self):
        driver = FakeDriver()
        _, sessions = make_library([driver])
        with pytest.raises(ValueError):
            sessions.wait_for_page_loaded(state="nonsense")

    def test_wait_for_page_loaded_tolerates_transient_driver_error(self):
        driver = FakeDriver()
        probes = {"n": 0}
        original = driver.execute_script

        def flaky_script(script, *args):
            probes["n"] += 1
            if probes["n"] == 1:
                raise WebDriverException("navigating")
            return original(script, *args)

        driver.execute_script = flaky_script
        _, sessions = make_library([driver])
        assert sessions.wait_for_page_loaded(
            timeout=1, poll_interval=0.05
        ) == "complete"

    def test_go_to_safely_retries_and_returns_url(self):
        driver = FakeDriver()
        library, sessions = make_library([driver])
        calls = {"n": 0}

        original_go_to = library.browser_management.go_to

        def flaky_go_to(url):
            calls["n"] += 1
            if calls["n"] == 1:
                raise WebDriverException("navigation interrupted")
            return original_go_to(url)

        library.go_to = flaky_go_to
        result = sessions.go_to_safely(
            "https://example.test/page", retries=3, backoff=0.01
        )
        assert result == driver.url
        assert calls["n"] == 2
        assert library.browser_management.navigations == [
            "https://example.test/page"
        ]


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def test_session_keywords_registered_on_library():
    library = Selenium()
    try:
        assert isinstance(library.sessions, SessionKeywords)
        names = library.get_keyword_names()
        # Outside a running Robot session the library exposes the Python
        # method names; inside Robot they become Title Case keywords.
        for expected in [
            "save_browser_state",
            "restore_browser_state",
            "switch_to_new_window",
            "close_window_safely",
            "browser_window",
            "close_browser_safely",
            "close_all_browsers_safely",
            "prune_dead_browsers",
            "run_with_retry",
            "wait_for_page_loaded",
            "go_to_safely",
            "dismiss_any_alert",
        ]:
            assert expected in names, expected
        # Every method must be exposed as a runnable keyword.
        for method in [
            library.save_browser_state,
            library.close_all_browsers_safely,
            library.run_with_retry,
        ]:
            assert callable(method)
    finally:
        library.close_all_browsers_safely()


# --------------------------------------------------------------------------- #
# Real browser integration (skipped without Chrome + chromedriver)
# --------------------------------------------------------------------------- #
chrome_available = bool(
    shutil.which("google-chrome")
    or shutil.which("chrome")
    or shutil.which("chromium")
    or shutil.which("chromium-browser")
) and bool(shutil.which("chromedriver"))


@pytest.mark.skipif(
    not chrome_available, reason="Chrome/chromedriver not available"
)
class TestRealBrowserSessions:
    @pytest.fixture
    def library(self):
        lib = Selenium()
        yield lib
        lib.close_all_browsers_safely()

    def test_window_state_isolation_with_popup(self, library):
        library.open_available_browser(
            FIXTURE_PAGE, headless=True, browser_selection="Chrome"
        )
        main_handle = library.driver.current_window_handle
        before = library.get_window_handles()

        state = library.save_browser_state()
        library.click_element("id:open-popup")
        popup = library.switch_to_new_window(
            timeout=10, known_handles=before
        )
        assert popup != main_handle
        library.page_should_contain("popup content")

        # Restoring must bring us back to the main page even after the popup
        # work is done.
        library.restore_browser_state(state)
        assert library.driver.current_window_handle == main_handle
        library.page_should_contain("Main page")

    def test_browser_window_context_manager(self, library):
        library.open_available_browser(
            FIXTURE_PAGE, headless=True, browser_selection="Chrome"
        )
        main_handle = library.driver.current_window_handle
        # The snapshot must be taken before switching into the popup so the
        # scope can return to this exact window on exit.
        before = library.get_window_handles()
        library.click_element("id:open-popup")
        popup = library.switch_to_new_window(
            timeout=10, known_handles=before, switch=False
        )

        with library.browser_window(popup):
            assert library.driver.current_window_handle == popup
            library.page_should_contain("popup content")
        assert library.driver.current_window_handle == main_handle

    def test_two_browsers_keep_isolated_state(self, library):
        first = library.open_available_browser(
            "about:blank", headless=True, browser_selection="Chrome",
            alias="worker-1",
        )
        second = library.open_available_browser(
            "about:blank", headless=True, browser_selection="Chrome",
            alias="worker-2",
        )
        assert first != second

        library.switch_browser(first)
        state_one = library.save_browser_state()
        with library.browser_scope(second):
            library.go_to(FIXTURE_PAGE)
            library.wait_for_page_loaded(timeout=10)
            # The second browser navigated to the fixture page and shows its
            # real document title.
            assert library.get_title() == (
                "Popup and dynamic page fixture"
            )
        # The scope restored the first browser, whose blank page is untouched.
        assert library._drivers.current_index == first
        assert library.get_location() == "about:blank"
        # Explicit restore yields the same result and stays on browser one.
        assert library.restore_browser_state(state_one) is True
        assert library._drivers.current_index == first

    def test_retry_recovers_from_transient_failure(self, library):
        library.open_available_browser(
            FIXTURE_PAGE, headless=True, browser_selection="Chrome"
        )
        attempts = {"n": 0}

        def wait_for_dynamic_panel():
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise TimeoutException("panel not rendered yet")
            library.wait_until_element_is_visible("id:dynamic-panel", 5)

        library.run_with_retry(wait_for_dynamic_panel, retries=3, backoff=0.2)
        assert library.is_element_visible("id:dynamic-panel")

    def test_dismiss_any_alert_on_real_browser(self, library):
        library.open_available_browser(
            FIXTURE_PAGE, headless=True, browser_selection="Chrome"
        )
        library.click_element("id:open-alert")
        message = library.dismiss_any_alert(timeout=5)
        assert message == "blocking dialog"
        # The driver must remain usable after the dialog was handled.
        assert library.get_title() == "Popup and dynamic page fixture"

    def test_close_all_browsers_safely_leaves_no_sessions(self, library):
        library.open_available_browser(
            "about:blank", headless=True, browser_selection="Chrome"
        )
        library.open_available_browser(
            "about:blank", headless=True, browser_selection="Chrome"
        )
        assert len(library.get_browser_ids()) == 2
        closed = library.close_all_browsers_safely()
        assert closed == 2
        assert library.get_browser_ids() == []
