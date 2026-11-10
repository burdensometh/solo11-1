*** Settings ***
Library    Collections
Library    RPA.Browser.Selenium

# These tasks need a real browser; the Python suite
# (test_browser_sessions.py) covers the same isolation/cleanup/retry logic
# with fake drivers and is always runnable. The whole file is skipped when
# Chrome is unavailable on the host.
Suite Setup      Setup Browser Or Skip
Suite Teardown   Close All Browsers Safely

Default Tags     rpa.browser.sessions

*** Variables ***
${FIXTURE}       file://${CURDIR}${/}..${/}resources${/}browser_sessions_fixture.html


*** Keywords ***
Setup Browser Or Skip
    ${has_chrome}=    Evaluate    __import__("shutil").which("google-chrome") or __import__("shutil").which("chrome") or __import__("shutil").which("chromium") or __import__("shutil").which("chromium-browser")
    ${has_driver}=    Evaluate    __import__("shutil").which("chromedriver")
    Run Keyword If    not ($has_chrome and $has_driver)
    ...    Skip    Chrome/chromedriver not available
    Open Available Browser    about:blank    headless=${True}    browser_selection=Chrome

Flaky Dynamic Panel Check
    [Arguments]    ${attempts}
    ${value}=    Evaluate    int($attempts[0]) + 1
    Set List Value    ${attempts}    0    ${value}
    Run Keyword If    ${value} == 1    Fail    panel not rendered yet
    Wait Until Element Is Visible    id:dynamic-panel    timeout=10s


Close Extra Windows
    [Arguments]    ${keep_handle}
    @{all_windows}=    Get Window Handles
    FOR    ${handle}    IN    @{all_windows}
        Run Keyword If    '${handle}' != '${keep_handle}'    Close Window Safely    ${handle}
    END


*** Tasks ***
Popup Window State Is Restored
    ${state}=    Save Browser State
    @{before}=    Get Window Handles
    Go To    ${FIXTURE}
    Click Element    id:open-popup
    ${popup}=    Switch To New Window    timeout=10s    known_handles=${before}
    Should Not Be Equal    ${popup}    ${state.handle}
    Page Should Contain    popup content
    Restore Browser State    ${state}
    Page Should Contain    Main page

Explicit Save And Restore Around Popup
    Go To    ${FIXTURE}
    # A previous task may have left a popup open while restoring the main
    # window; close every extra window and pin the remaining main window.
    @{open_windows}=    Get Window Handles
    Close Extra Windows    ${open_windows}[0]
    Switch Window    ${open_windows}[0]
    # The snapshot is taken on the main window BEFORE switching into the
    # popup, so restoration returns to this exact window.
    ${state}=    Save Browser State
    @{before}=    Get Window Handles
    Click Element    id:open-popup
    ${popup}=    Switch To New Window    timeout=10s    known_handles=${before}
    Switch Window    ${popup}
    Page Should Contain    popup content
    Restore Browser State    ${state}
    Page Should Contain    Main page
    # We are back on the main window; the popup still exists as second window.
    @{current}=    Get Window Handles
    Length Should Be    ${current}    2

Alert Is Dismissed Without Failing
    Go To    ${FIXTURE}
    Click Element    id:open-alert
    ${message}=    Dismiss Any Alert    timeout=5s
    Should Be Equal    ${message}    blocking dialog
    ${title}=    Get Title
    Should Not Be Empty    ${title}

Go To Safely Handles Dynamic Page
    ${url}=    Go To Safely    ${FIXTURE}    retries=2    backoff=0.1s
    Should Contain    ${url}    browser_sessions_fixture.html
    Wait For Page Loaded    state=complete    timeout=10s
    Wait Until Element Is Visible    id:dynamic-panel    timeout=10s
    Page Should Contain    rendered

Run With Retry Recovers From Transient Failure
    Go To    ${FIXTURE}
    @{attempts}=    Create List    0
    Run With Retry    Flaky Dynamic Panel Check    ${attempts}    retries=3    backoff=0.2s
    Page Should Contain    rendered

Safe Cleanup Closes Every Browser
    Open Available Browser    about:blank    headless=${True}    browser_selection=Chrome
    Open Available Browser    about:blank    headless=${True}    browser_selection=Chrome
    @{ids}=    Get Browser Ids
    Length Should Be    ${ids}    3
    ${closed}=    Close All Browsers Safely
    Should Be Equal As Integers    ${closed}    3
    @{remaining}=    Get Browser Ids
    Length Should Be    ${remaining}    0
