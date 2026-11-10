"""Isolated multi-window session management for `RPA.Browser.Selenium`.

This module adds long-running, state-isolated browser session support on top
of SeleniumLibrary without changing any existing keyword behaviour:

- browser/window state snapshots and guaranteed restoration
  (`save_browser_state` / `restore_browser_state` / `browser_window` /
  `browser_scope`);
- reliable popup/window discovery and switching that survives interleaved
  navigations and alert dialogs (`switch_to_new_window`, `dismiss_any_alert`,
  `close_window_safely`);
- failure-aware cleanup that never leaves half-finished browsers behind
  (`close_browser_safely`, `close_all_browsers_safely`,
  `prune_dead_browsers`);
- retry support for flaky, dynamically rendered pages
  (`run_with_retry`, `go_to_safely`, `wait_for_page_loaded`).

Everything here is additive: existing keywords keep working exactly as
before and the new ones are opt-in.
"""

import contextlib
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from selenium.common.exceptions import (
    InvalidSelectorException,
    NoAlertPresentException,
    NoSuchElementException,
    NoSuchWindowException,
    TimeoutException,
    WebDriverException,
)
from SeleniumLibrary.base import keyword
from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


logger = logging.getLogger(__name__)

# Selenium errors caused by a bad locator/argument rather than transient page
# state; retrying them would only waste time.
NON_RETRYABLE_EXCEPTIONS: Tuple[type, ...] = (
    InvalidSelectorException,
    NoSuchElementException,
    ValueError,
    TypeError,
)

_READY_STATES = {"complete", "interactive", "loading"}


def _is_retryable(exc: BaseException) -> bool:
    """Return whether ``exc`` represents a transient browser failure."""
    if isinstance(exc, NON_RETRYABLE_EXCEPTIONS):
        return False
    return isinstance(exc, Exception)


def _log_retry(retry_state: RetryCallState) -> None:
    """Log every failed attempt before tenacity sleeps and retries."""
    if not retry_state.outcome:
        return
    exc = retry_state.outcome.exception()
    logger.warning(
        "Browser action attempt %s failed (%s); %s",
        retry_state.attempt_number,
        exc,
        "retrying" if retry_state.attempt_number < 10 else "giving up",
    )


@dataclass
class BrowserState:
    """Snapshot of everything identifying where a browser session is.

    ``index`` is the 1-based SeleniumLibrary connection index and ``handle``
    is the window/tab handle. ``frame_chain`` and ``url`` are retained for
    diagnostics; frame restoration resets to the document's default content
    because Selenium does not expose the active frame chain portably.
    """

    index: Optional[int]
    handle: Optional[str]
    frame_chain: List[Any] = field(default_factory=list)
    url: Optional[str] = None

    def restore(self, driver) -> None:
        """Switch ``driver`` back to the captured window and top frame."""
        driver.switch_to.window(self.handle)
        driver.switch_to.default_content()
        for frame in self.frame_chain:
            driver.switch_to.frame(frame)


def _current_frame_chain(driver) -> List[Any]:
    """Best-effort capture of the active frame nesting.

    Selenium does not expose the active frame chain in a driver-agnostic way,
    so this currently returns an empty list (restoration therefore resets to
    default content). The hook is kept so richer drivers can populate it later
    without changing the snapshot/restore public API, and it never raises.
    """
    del driver
    return []


class SessionKeywords:
    """Library component exposing session isolation keywords.

    The component is constructed with the owning :class:`Selenium` library as
    ``ctx`` and reuses its SeleniumLibrary connection cache and keyword
    implementations instead of keeping browser state of its own; all scopes
    carry their snapshots on the stack of the ``with`` block that created
    them.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.logger = logging.getLogger("RPA.Browser.Selenium")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @property
    def _drivers(self):
        # SeleniumLibrary connection cache (WebDriverCache instance).
        return self.ctx._drivers  # pylint: disable=protected-access

    def _require_driver(self):
        driver = self._drivers.current
        if not driver:
            raise RuntimeError(
                "No open browser; use 'Open Available Browser' first."
            )
        return driver

    def _snapshot(self) -> BrowserState:
        driver = self._require_driver()
        handle = None
        url = None
        with contextlib.suppress(WebDriverException):
            handle = driver.current_window_handle
        with contextlib.suppress(WebDriverException):
            url = driver.current_url
        return BrowserState(
            index=self._drivers.current_index,
            handle=handle,
            frame_chain=_current_frame_chain(driver),
            url=url,
        )

    def _restore(self, state: BrowserState) -> bool:
        """Restore a previously saved state.

        Returns ``True`` on success. A missing browser connection or a window
        that was closed in the meantime is reported as ``False`` instead of
        raising so cleanup paths stay safe.
        """
        if state.index is None:
            return False
        try:
            self._drivers.switch(state.index)
        except RuntimeError:
            self.logger.warning(
                "Cannot restore browser index %s: connection no longer exists",
                state.index,
            )
            return False
        driver = self._drivers.current
        if not state.handle:
            return True
        try:
            if state.handle not in driver.window_handles:
                self.logger.warning(
                    "Cannot restore window %s: it was closed; staying on %s",
                    state.handle,
                    self._safe_handle(driver),
                )
                return False
            state.restore(driver)
        except WebDriverException as exc:
            self.logger.warning("Failed to restore browser state: %s", exc)
            return False
        return True

    @staticmethod
    def _safe_handle(driver) -> Optional[str]:
        with contextlib.suppress(WebDriverException):
            return driver.current_window_handle
        return None

    # ------------------------------------------------------------------ #
    # Alert handling
    # ------------------------------------------------------------------ #
    @keyword
    def dismiss_any_alert(
        self, action: str = "ACCEPT", timeout: Any = None
    ) -> Optional[str]:
        """Handle an alert/confirm/prompt dialog if one is currently present.

        Unlike `Handle Alert`, this keyword never fails when no alert is open,
        which makes it safe to call inside cleanup and retry paths where an
        unexpected dialog would otherwise block every subsequent command.

        ``action`` is one of ``ACCEPT`` (default), ``DISMISS`` or ``LEAVE``.

        ``timeout`` sets how long to wait for an alert to appear; by default
        the check is immediate (a short 0.2 s grace period covers an alert
        still being rendered).

        Returns the alert message as a string, or ``None`` when no alert was
        present.

        Example:

        | ${msg}= | Dismiss Any Alert | action=DISMISS |
        """
        driver = self._require_driver()
        deadline = time.time() + self._alert_timeout(timeout)
        while True:
            try:
                alert = driver.switch_to.alert
                text = " ".join((alert.text or "").splitlines())
                normalized = action.upper()
                if normalized == "ACCEPT":
                    alert.accept()
                elif normalized == "DISMISS":
                    alert.dismiss()
                elif normalized == "LEAVE":
                    pass
                else:
                    raise ValueError(
                        f"Invalid alert action {action!r}; use ACCEPT, DISMISS"
                        " or LEAVE."
                    )
                self.logger.debug("Dismissed alert with message: %s", text)
                return text
            except NoAlertPresentException:
                if time.time() >= deadline:
                    return None
                time.sleep(0.1)
            except WebDriverException as exc:
                # A closed/switching window can surface generic driver errors
                # while probing for an alert; treat them like "no alert".
                self.logger.debug("Alert probe failed: %s", exc)
                return None

    def _alert_timeout(self, timeout) -> float:
        if timeout is None:
            return 0.2
        if isinstance(timeout, (int, float)):
            return float(timeout)
        try:
            return float(self.ctx.browser_management.get_timeout(timeout))
        except Exception:  # pylint: disable=broad-except
            return 0.2

    # ------------------------------------------------------------------ #
    # State snapshots
    # ------------------------------------------------------------------ #
    @keyword
    def save_browser_state(self) -> BrowserState:
        """Capture the active browser, window and frame selection.

        The returned opaque :class:`BrowserState` object can be stored in a
        variable and later passed to `Restore Browser State`, guaranteeing
        that operations which open popups or switch windows do not leak their
        window selection into the caller's context.

        Example:

        | ${state}= | Save Browser State |
        | Switch To New Window |
        | ... |
        | Restore Browser State | ${state} |
        """
        return self._snapshot()

    @keyword
    def restore_browser_state(self, state: BrowserState) -> bool:
        """Switch back to the browser/window captured in ``state``.

        ``state`` must be an object returned by `Save Browser State`. Returns
        ``True`` when the state was fully restored and ``False`` when the
        target browser or window had already been closed (the keyword logs a
        warning instead of failing in that case).

        Any alert blocking the driver is accepted first, because Selenium
        cannot switch windows while a modal dialog is open.
        """
        if self._drivers.current:
            self.dismiss_any_alert()
        return self._restore(state)

    # ------------------------------------------------------------------ #
    # Window isolation
    # ------------------------------------------------------------------ #
    @keyword
    def switch_to_new_window(
        self,
        timeout: Any = None,
        known_handles: Optional[List[str]] = None,
        switch: bool = True,  # pylint: disable=redefined-builtin
    ) -> str:
        """Wait for a newly opened browser window/tab and optionally switch.

        Popup windows opened by clicks or JavaScript are discovered by
        diffing the current window handles against ``known_handles`` (the
        handles captured before the action that opened the popup). When
        ``known_handles`` is not provided, any handle other than the current
        one is considered new.

        ``timeout`` maximum wait time (library default timeout when omitted).

        ``switch`` when ``True`` (default) selects the new window; the
        previous state is still restorable with `Save Browser State` /
        `Browser Window`. Set to ``False`` to only return the handle.

        Returns the new window handle. Fails with a Selenium
        ``TimeoutException`` if no new window appears in time.

        Example:

        | ${before}= | Get Window Handles |
        | Click Element | id:open-popup |
        | ${handle}= | Switch To New Window | known_handles=${before} |
        """
        driver = self._require_driver()
        wait_seconds = float(
            self.ctx.browser_management.get_timeout(timeout)
        )
        end_time = time.time() + wait_seconds
        previous = set(known_handles) if known_handles is not None else set()
        new_handle: Optional[str] = None
        while time.time() <= end_time:
            try:
                handles = list(driver.window_handles)
            except WebDriverException as exc:
                self.logger.debug("Window enumeration failed: %s", exc)
                time.sleep(0.25)
                continue
            candidates = [
                handle
                for handle in handles
                if known_handles is None
                or handle not in previous
            ]
            current = self._safe_handle(driver)
            if known_handles is None and current in candidates:
                candidates.remove(current)
            if candidates:
                new_handle = candidates[0]
                break
            time.sleep(0.25)
        if new_handle is None:
            raise TimeoutException(
                f"No new browser window appeared within {wait_seconds}s"
            )
        if switch:
            self.dismiss_any_alert()
            driver.switch_to.window(new_handle)
        return new_handle

    @keyword
    def close_window_safely(
        self,
        handle: Optional[str] = None,
        restore_state: Optional[BrowserState] = None,
    ) -> bool:
        """Close a window/tab without ever failing the cleanup path.

        ``handle`` identifies the window to close; when omitted the currently
        selected window is closed. An open alert is dismissed first so the
        close command cannot get blocked by a modal dialog.

        ``restore_state`` optionally re-applies a state previously captured
        with `Save Browser State` after closing. Closing the last window of
        a browser quits that browser session instead of erroring.

        Returns ``True`` if a window was closed, ``False`` if there was
        nothing to close. All driver errors are logged and swallowed, which
        makes this keyword suitable for ``[Teardown]`` usage.

        Example:

        | Close Window Safely | ${popup_handle} | restore_state=${state} |
        """
        driver = self._require_driver()
        target = handle or self._safe_handle(driver)
        if target is None:
            return False
        try:
            if target != self._safe_handle(driver):
                driver.switch_to.window(target)
            self.dismiss_any_alert()
            handles = list(driver.window_handles)
            if len(handles) <= 1:
                # Closing the final window terminates the whole browser; do
                # it through the cache so its bookkeeping stays consistent.
                self.close_browser_safely()
                return True
            driver.close()
            remaining = [item for item in driver.window_handles if item != target]
            if remaining:
                driver.switch_to.window(remaining[0])
        except NoSuchWindowException:
            self.logger.debug("Window %s was already closed", target)
        except WebDriverException as exc:
            self.logger.warning("Error while closing window %s: %s", target, exc)
            return False
        if restore_state is not None:
            self._restore(restore_state)
        return True

    @contextlib.contextmanager
    def window_scope(self, handle: str):
        """Python context manager isolating work to ``handle``.

        On exit the previously selected browser/window is restored even when
        the block raised, and an alert left behind by the block is accepted.
        """
        state = self._snapshot()
        driver = self._require_driver()
        driver.switch_to.window(handle)
        try:
            yield
        finally:
            self.dismiss_any_alert()
            self._restore(state)

    @keyword
    def browser_window(
        self,
        handle: Optional[str] = None,
        locator: Optional[str] = None,
        timeout: Any = None,
    ) -> Any:
        """Run Robot Framework keywords inside an isolated window scope.

        The active browser/window selection is saved on entry and guaranteed
        to be restored on exit (including failures), so popup-driven flows
        cannot leak their window state back into the caller. Browser
        selection (alias/index) is restored as well, keeping parallel
        browser instances isolated.

        Provide either a window ``handle`` (as returned by
        `Switch To New Window` / `Get Window Handles`) or a SeleniumLibrary
        window ``locator`` (title/url/name strategy, see `Switch Window`).

        Returns a context manager usable in two equivalent ways.

        Robot Framework (explicit save/restore inside ``TRY/FINALLY``)::

            ${state}=    Save Browser State
            @{handles}=    Get Window Handles
            Click Element    id:open-popup
            Switch To New Window    known_handles=${handles}
            TRY
            ...    Page Should Contain    Thank you
            ...    FINALLY
            ...    Restore Browser State    ${state}
            ...    END

        Python (the returned object is a context manager)::

            with lib.browser_window(popup_handle):
                lib.page_should_contain("Thank you")
        """
        return self._WindowScope(self, handle=handle, locator=locator, timeout=timeout)

    class _WindowScope:
        """Context manager backing the `Browser Window` keyword."""

        def __init__(self, owner, handle, locator, timeout):
            self._owner = owner
            self._handle = handle
            self._locator = locator
            self._timeout = timeout
            self._state: Optional[BrowserState] = None

        def __enter__(self):
            self._state = self._owner._snapshot()  # pylint: disable=protected-access
            driver = self._owner._require_driver()  # pylint: disable=protected-access
            if self._handle is not None:
                driver.switch_to.window(self._handle)
            elif self._locator is not None:
                # Reuse SeleniumLibrary's own window selection strategies.
                self._owner.ctx.switch_window(
                    self._locator, timeout=self._timeout
                )
            else:
                raise ValueError(
                    "Provide either a window handle or a window locator."
                )
            return self

        def __exit__(self, exc_type, exc, tb):
            # Never let restoration mask the original exception.
            self._owner.dismiss_any_alert()
            self._owner._restore(self._state)  # pylint: disable=protected-access
            return False

    # ------------------------------------------------------------------ #
    # Browser-level scopes
    # ------------------------------------------------------------------ #
    @keyword
    @contextlib.contextmanager
    def browser_scope(self, alias_or_index: Any):
        """Python context manager isolating work to one browser instance.

        Temporarily selects the browser identified by ``alias_or_index``
        (an alias string or 1-based index) and restores the previously
        active browser on exit, even when the block fails. This is the
        Python-level counterpart of `Save Browser State` /
        `Restore Browser State` and prevents interleaved multi-browser flows
        from sharing each other's page state.

        Registered as a keyword so it is reachable from the library object in
        plain Python as well as from Robot Framework; it is a context
        manager and is intended for ``with`` usage.
        """
        state = self._snapshot()
        self._drivers.switch(alias_or_index)
        try:
            yield
        finally:
            self.dismiss_any_alert()
            self._restore(state)

    # ------------------------------------------------------------------ #
    # Reliable lifecycle / cleanup
    # ------------------------------------------------------------------ #
    @keyword
    def close_browser_safely(self, alias_or_index: Any = None) -> bool:
        """Quit one browser, swallowing every cleanup-time failure.

        Targets the current browser unless ``alias_or_index`` (alias string
        or 1-based index) is given. A blocking alert is dismissed first and
        a dead/unresponsive driver is force-stopped through its service
        process, so no orphaned browser or webdriver process is left behind.

        Returns ``True`` when a browser was closed and ``False`` when there
        was no matching browser to close.

        Example:

        | Close Browser Safely | alias=Worker-3 |
        """
        cache = self._drivers
        if alias_or_index is not None:
            index = cache.get_index(alias_or_index)
            if index is None:
                self.logger.warning(
                    "Cannot close browser %r: no such open browser",
                    alias_or_index,
                )
                return False
            target_driver = cache.get_connection(index)
            was_current = cache.current_index == index
            state = self._snapshot() if cache.current and not was_current else None
            if not was_current:
                try:
                    cache.switch(index)
                except RuntimeError:
                    return False
        else:
            if not cache.current:
                return False
            target_driver = cache.current
            was_current = True
            index = cache.current_index
            state = None
        try:
            with contextlib.suppress(WebDriverException):
                target_driver.switch_to.alert.accept()
            target_driver.quit()
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning(
                "Graceful browser quit failed (%s); force-stopping service",
                exc,
            )
            self._force_stop(target_driver)
        finally:
            # Keep the SeleniumLibrary cache bookkeeping in sync.
            with contextlib.suppress(Exception):
                cache._closed.add(target_driver)  # pylint: disable=protected-access
                if was_current:
                    # Nothing to select anymore: prevent later keywords from
                    # targeting the just-closed driver.
                    no_current = cache._no_current  # pylint: disable=protected-access
                    cache.current = no_current
                    cache.current_index = None
            if state is not None:
                self._restore(state)
        return True

    @staticmethod
    def _force_stop(driver) -> None:
        """Last-resort cleanup: kill the webdriver service process."""
        service = getattr(driver, "service", None)
        process = getattr(service, "process", None)
        if process is not None:
            with contextlib.suppress(Exception):
                process.kill()
        with contextlib.suppress(Exception):
            service.stop()

    @keyword
    def close_all_browsers_safely(self) -> int:
        """Close every open browser, never raising and leaving no orphans.

        Every registered browser is quit independently; a failure while
        closing one browser does not prevent the others from being cleaned
        up, and unresponsive drivers have their service process killed.
        The SeleniumLibrary connection cache is reset afterwards.

        Returns the number of browser sessions that were closed.
        """
        cache = self._drivers
        drivers = list(cache._connections)  # pylint: disable=protected-access
        closed = 0
        for driver in drivers:
            if driver in getattr(cache, "_closed", set()):
                continue
            try:
                with contextlib.suppress(WebDriverException):
                    driver.switch_to.alert.accept()
                driver.quit()
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning(
                    "Error while quitting browser during cleanup: %s", exc
                )
                self._force_stop(driver)
            closed += 1
        with contextlib.suppress(Exception):
            cache.empty_cache()
        if closed:
            self.logger.info("Safely closed %s browser(s)", closed)
        return closed

    @keyword
    def prune_dead_browsers(self) -> List[int]:
        """Remove browser sessions whose webdriver connection has died.

        Long-running flows eventually hit crashed tabs, network drops or
        killed driver processes. This probes every registered session with
        a cheap command, force-stops the ones that no longer respond and
        drops them from the connection cache so subsequent keywords target
        live browsers only.

        Returns the 1-based indexes of the pruned browser sessions.
        """
        cache = self._drivers
        dead: List[Any] = []
        for driver in list(cache._connections):  # pylint: disable=protected-access
            if driver in getattr(cache, "_closed", set()):
                continue
            try:
                driver.current_url
            except WebDriverException as exc:
                self.logger.warning(
                    "Browser session %s is unresponsive (%s); pruning",
                    getattr(driver, "session_id", "?"),
                    exc,
                )
                self._force_stop(driver)
                dead.append(driver)
        pruned_indexes: List[int] = []
        for driver in dead:
            connections = cache._connections  # pylint: disable=protected-access
            if driver in connections:
                pruned_indexes.append(connections.index(driver) + 1)
            cache._closed.add(driver)  # pylint: disable=protected-access
        if pruned_indexes:
            self.logger.info("Pruned dead browser sessions: %s", pruned_indexes)
        return pruned_indexes

    # ------------------------------------------------------------------ #
    # Retry
    # ------------------------------------------------------------------ #
    def _seconds(self, value: Any, default: float) -> float:
        """Convert a numeric or Robot time-string value to seconds."""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(self.ctx.browser_management.get_timeout(value))
        except Exception:  # pylint: disable=broad-except
            return float(value)

    @keyword
    def run_with_retry(
        self,
        action: Callable,
        *args,
        retries: Any = 3,
        backoff: Any = 1.0,
        max_backoff: Any = 10.0,
        retry_on: Optional[Any] = None,
        cleanup: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """Run ``action`` with retries, exponential backoff and cleanup.

        Designed for flaky dynamic pages: transient Selenium errors
        (timeouts, stale elements, detached windows, crashed target pages)
        trigger a retry while programming errors such as invalid selectors
        fail immediately without wasting attempts.

        ``action`` any callable; extra positional/keyword arguments are
        forwarded to it. A Robot Framework keyword name (string) can also be
        passed and is resolved through the running Robot session.

        ``retries`` total number of attempts (default 3).

        ``backoff`` / ``max_backoff`` exponential wait parameters in seconds;
        Robot time strings such as ``0.5s`` are accepted too. The wait before
        attempt *n* is ``min(backoff * 2**(n-1), max_backoff)``.

        ``retry_on`` optional exception type or tuple of types that should
        trigger a retry; when omitted all retryable Selenium exceptions are
        retried and selector/argument errors are not.

        ``cleanup`` optional callable executed (and logged if it fails)
        before every retry, e.g. dismissing an alert or closing a leftover
        popup, so each retry starts from a known state.

        Returns the value produced by ``action``. Re-raises the last
        exception if every attempt fails.

        Python example::

            lib.run_with_retry(
                lib.wait_until_element_is_visible,
                "id:dynamic-panel",
                retries=5,
                cleanup=lambda: lib.dismiss_any_alert(),
            )
        """
        target, resolved_args, resolved_kwargs = self._resolve_action(
            action, args, kwargs
        )
        attempts = max(1, int(retries))
        backoff_seconds = self._seconds(backoff, 1.0)
        max_backoff_seconds = self._seconds(max_backoff, 10.0)

        def predicate(exc: BaseException) -> bool:
            if retry_on is not None:
                return isinstance(exc, retry_on)
            return _is_retryable(exc)

        retryer = Retrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(
                multiplier=backoff_seconds,
                max=max_backoff_seconds,
                exp_base=2,
            ),
            retry=retry_if_exception(predicate),
            before_sleep=_log_retry,
            reraise=True,
        )
        attempt = {"number": 0}

        @functools.wraps(target)
        def invoke():
            attempt["number"] += 1
            if attempt["number"] > 1 and cleanup is not None:
                self._run_cleanup(cleanup)
            return target(*resolved_args, **resolved_kwargs)

        try:
            return retryer(invoke)
        except RetryError as exc:  # pragma: no cover - reraise=True normally
            raise exc.last_attempt.exception() from exc

    def _resolve_action(self, action, args, kwargs):
        if callable(action):
            return action, list(args), dict(kwargs)
        if isinstance(action, str):
            from robot.libraries.BuiltIn import (  # local import: RF optional
                BuiltIn,
                RobotNotRunningError,
            )

            keyword_name = action

            def runner(*inner_args, **inner_kwargs):
                try:
                    return BuiltIn().run_keyword(
                        keyword_name, *inner_args, **inner_kwargs
                    )
                except RobotNotRunningError as exc:
                    raise RuntimeError(
                        f"Cannot run keyword {keyword_name!r} outside a Robot"
                        " Framework run; pass a callable instead."
                    ) from exc

            return runner, list(args), dict(kwargs)
        raise TypeError(
            f"action must be callable or a keyword name, got {type(action)!r}"
        )

    def _run_cleanup(self, cleanup: Callable) -> None:
        try:
            cleanup()
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("Retry cleanup action failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Navigation / dynamic page readiness
    # ------------------------------------------------------------------ #
    @keyword
    def wait_for_page_loaded(
        self,
        state: str = "complete",
        timeout: Any = None,
        poll_interval: float = 0.25,
    ) -> str:
        """Wait until the page reaches the requested ``document.readyState``.

        Works for both server-rendered and dynamic (JS-heavy) pages because
        it polls the browser's own readiness signal instead of guessing from
        element timing. ``state`` is one of:

        - ``loading``: document still parsing;
        - ``interactive``: DOM ready, resources may still load;
        - ``complete`` (default): everything including sub-resources loaded.

        ``timeout`` uses the library default when omitted.
        ``poll_interval`` sets the polling cadence in seconds.

        Returns the final ready state string. Raises Selenium
        ``TimeoutException`` when the state is not reached in time.
        """
        normalized = state.lower()
        if normalized not in _READY_STATES:
            raise ValueError(
                f"Invalid ready state {state!r}; expected one of "
                f"{sorted(_READY_STATES)}."
            )
        driver = self._require_driver()
        wait_seconds = float(
            self.ctx.browser_management.get_timeout(timeout)
        )
        end_time = time.time() + wait_seconds
        readiness_rank = {"loading": 0, "interactive": 1, "complete": 2}
        required = readiness_rank[normalized]
        current = None
        while time.time() <= end_time:
            try:
                current = str(
                    driver.execute_script("return document.readyState;")
                ).lower()
            except WebDriverException as exc:
                # Navigation in progress; the next poll picks the new page.
                self.logger.debug("readyState probe failed: %s", exc)
                time.sleep(poll_interval)
                continue
            if readiness_rank.get(current, -1) >= required:
                return current
            time.sleep(poll_interval)
        raise TimeoutException(
            f"Page did not reach readyState {normalized!r} within "
            f"{wait_seconds}s (last state: {current!r})"
        )

    @keyword
    def go_to_safely(
        self,
        url: str,
        retries: Any = 3,
        backoff: Any = 1.0,
        max_backoff: Any = 10.0,
        timeout: Any = None,
        wait_state: str = "complete",
        dismiss_alerts: bool = True,
    ) -> str:
        """Navigate to ``url`` with retry, alert handling and readiness wait.

        A hardened alternative to `Go To` for long-running flows that switch
        between ordinary pages and dynamic applications. Each attempt:

        1. dismisses any modal dialog left over from the previous page;
        2. performs the navigation (retried on transient driver errors);
        3. waits for the requested page ready state.

        ``retries`` total navigation attempts.
        ``backoff`` / ``max_backoff`` exponential retry wait in seconds.
        ``timeout`` ready-state wait per attempt (library default if omitted).
        ``wait_state`` target ``document.readyState`` — set to
        ``interactive`` for SPAs that never reach ``complete`` quickly.
        ``dismiss_alerts`` accept any blocking alert before navigating.

        Returns the final URL of the page. The library's default URL scheme
        handling still applies, so existing ``Go To`` usage is unchanged when
        this keyword is not used.
        """
        def navigate():
            if dismiss_alerts:
                self.dismiss_any_alert()
            self.ctx.go_to(url)
            return self.wait_for_page_loaded(state=wait_state, timeout=timeout)

        self.run_with_retry(
            navigate,
            retries=retries,
            backoff=backoff,
            max_backoff=max_backoff,
            cleanup=lambda: self.dismiss_any_alert() if dismiss_alerts else None,
        )
        return self._require_driver().current_url
