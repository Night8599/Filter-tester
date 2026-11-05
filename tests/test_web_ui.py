"""End-to-end test of the web interface in headless Chrome, against web/dev_server.py.

Skipped when Playwright or a Chrome/Chromium browser is not available.
Set FILTER_TESTER_SCREENSHOTS=<dir> to save screenshots of every tab.
"""

import os
import shutil
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
import dev_server  # noqa: E402

CHROME = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
SHOTS = os.environ.get("FILTER_TESTER_SCREENSHOTS")


@pytest.fixture(scope="module")
def server_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), dev_server.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/?backend=server"
    server.shutdown()


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        try:
            b = p.chromium.launch(executable_path=CHROME) if CHROME else p.chromium.launch()
        except Exception as err:  # no browser installed
            pytest.skip(f"no browser available: {err}")
        yield b
        b.close()


@pytest.fixture
def page(browser, server_url):
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(server_url)
    page.wait_for_function("document.querySelector('#status').textContent === 'Ready'", timeout=15000)
    yield page
    assert errors == [], errors
    page.close()


def shot(page, name):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(SHOTS) / f"{name}.png"), full_page=True)


def set_number(page, scope, label, value):
    row = page.locator(f"{scope} .param", has=page.locator(f"label:text-is('{label}')")).first
    row.locator("input[type=number]").fill(str(value))
    row.locator("input[type=number]").dispatch_event("change")
    return row


def wait_finished(page, timeout=60000):
    page.wait_for_function("document.querySelector('#status').textContent === 'Finished'", timeout=timeout)


def test_setup_tab(page):
    options = page.locator("#scenario option").all_inner_texts()
    assert options == [
        "Constant velocity",
        "Ornstein–Uhlenbeck process",
        "Mass–spring–damper",
        "Pendulum (nonlinear)",
        "Lorenz system (chaotic)",
    ]
    types = page.locator("#filter-type option").all_inner_texts()
    assert types == [
        "Kalman filter (KF)",
        "Extended Kalman filter (EKF)",
        "Unscented Kalman filter (UKF)",
        "Particle filter (PF)",
    ]
    assert page.locator(".filter-card").count() == 1
    assert page.locator("#system-params .param").count() > 5
    page.click("#add-filter")
    page.wait_for_function("document.querySelectorAll('.filter-card').length === 2")
    row = set_number(page, ".filter-card:nth-child(2)", "GPS noise variance R", 2.0)
    assert "changed" in row.get_attribute("class")
    # The slider follows the typed value (log slider: log10(2)).
    assert abs(float(row.locator("input[type=range]").input_value()) - 0.30103) < 0.01
    page.select_option("#scenario", "pendulum")
    page.wait_for_function("document.querySelectorAll('.filter-card').length === 1")
    assert "small-angle" in page.locator(".filter-card .type").inner_text()
    shot(page, "setup")


def test_single_run(page):
    set_number(page, "#system-params", "Duration [s]", 8)
    page.click("#add-filter")
    page.wait_for_function("document.querySelectorAll('.filter-card').length === 2")
    set_number(page, ".filter-card:nth-child(2)", "GPS noise variance R", 0.01)
    page.select_option("#speed", "0")
    page.click("#start")
    wait_finished(page)
    assert page.locator("#tab-simulation").is_visible()
    assert page.locator("#sim-single canvas").count() == 2  # position, velocity
    assert page.locator("#progress-text").inner_text().startswith("t = 8")
    shot(page, "simulation-single")
    page.click("button[data-tab=metrics]")
    rows = page.locator("#metrics-table tbody tr")
    assert rows.count() == 2
    assert "optimistic" in rows.nth(1).inner_text()  # R 25x too small
    assert page.locator("#meanvar-table tbody tr").count() == 3  # truth + 2 filters
    assert page.locator("#metrics-charts canvas").count() == 2  # variance per state
    shot(page, "metrics-single")


def test_monte_carlo(page):
    page.select_option("#scenario", "mass_spring")
    page.wait_for_function("document.querySelector('.filter-card input').value === 'KF (matched)'")
    set_number(page, "#system-params", "Duration [s]", 5)
    page.click("#add-filter")
    page.wait_for_function("document.querySelectorAll('.filter-card').length === 2")
    set_number(page, ".filter-card:nth-child(2)", "Model frequency ω [rad/s]", 4.0)
    page.check("input[name=mode][value=montecarlo]")
    assert page.locator("#runs").is_visible() and not page.locator("#speed").is_visible()
    page.fill("#runs", "8")
    page.click("#start")
    wait_finished(page)
    assert page.locator("#progress-text").inner_text() == "8 / 8 runs"
    shot(page, "simulation-montecarlo")
    page.click("button[data-tab=metrics]")
    header = page.locator("#metrics-table thead").inner_text()
    assert "ANEES" in header and "ANIS" in header
    assert page.locator("#metrics-table td.bad").count() >= 1  # wrong frequency is inconsistent
    assert page.locator("#meanvar-table tbody tr").count() == 2
    assert page.locator("#metrics-charts canvas").count() == 3  # RMSE boxplots: 2 states + total
    shot(page, "metrics-montecarlo")


def test_stop_and_validation(page):
    page.fill(".filter-card input[type=text]", "")
    page.click("#start")
    assert "name" in page.locator("#error").inner_text()
    page.fill(".filter-card input[type=text]", "KF")
    set_number(page, "#system-params", "Duration [s]", 200)
    page.select_option("#speed", "1")
    page.click("#start")
    page.wait_for_function("document.querySelector('#status').textContent.startsWith('Running')")
    page.wait_for_timeout(600)
    assert page.locator("#tab-simulation").is_visible()
    page.click("#stop-top")  # reachable from any tab while running
    page.wait_for_function("document.querySelector('#status').textContent === 'Stopped'")
    t_now = float(page.locator("#progress-text").inner_text().split()[2])
    assert 0 < t_now < 3  # paced at 1x real time
    assert page.locator("#start").is_enabled() and not page.locator("#stop-top").is_visible()


def test_nonlinear_filters_on_lorenz(page):
    page.select_option("#scenario", "lorenz")
    page.wait_for_function("document.querySelector('.filter-card input').value === 'EKF (matched)'")
    assert page.locator("#filter-type option").all_inner_texts() == [
        "Extended Kalman filter (EKF)",
        "Unscented Kalman filter (UKF)",
        "Particle filter (PF)",
    ]
    set_number(page, "#system-params", "Duration [s]", 3)
    for t in ("ukf", "pf"):
        page.select_option("#filter-type", t)
        page.click("#add-filter")
    page.wait_for_function("document.querySelectorAll('.filter-card').length === 3")
    names = [page.locator(".filter-card input[type=text]").nth(i).input_value() for i in range(3)]
    assert names == ["EKF (matched)", "UKF 1", "PF 1"]
    pf_card = page.locator(".filter-card").nth(2)
    assert pf_card.locator("label:text-is('Number of particles')").count() == 1
    assert page.locator(".filter-card").nth(1).locator("label:text-is('Spread α')").count() == 1
    set_number(page, ".filter-card:nth-child(3)", "Number of particles", 200)
    page.select_option("#speed", "0")
    page.click("#start")
    wait_finished(page)
    assert page.locator("#sim-single canvas").count() == 3  # x, y, z
    shot(page, "simulation-lorenz")
    page.click("button[data-tab=metrics]")
    rows = page.locator("#metrics-table tbody tr")
    assert [rows.nth(i).locator("td").first.inner_text() for i in range(3)] == names
    shot(page, "metrics-lorenz")
