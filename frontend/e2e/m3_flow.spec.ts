// M3 end-to-end flow (plan §16 M3: "play -> pause -> save -> reload browser/backend -> resume ->
// fork"), against the real backend, real Monza 2024 data and a real browser. The backend is
// killed abruptly in the middle (no clean shutdown) and the browser page is reloaded.

import { expect, test, type Page } from "@playwright/test";

import { Backend, E2E_BACKEND_PORT, hasMonzaData, MONZA_DATA } from "./backend";

const SAVE_NAME = "e2e checkpoint";

let backend: Backend;

test.beforeAll(async ({}, testInfo) => {
  // the save file lives in Playwright's output folder: kept for inspection after a failure,
  // wiped at the start of the next run (Windows keeps a killed process's files locked briefly)
  backend = new Backend(testInfo.outputPath("worlds.sqlite"), E2E_BACKEND_PORT);
});

test.afterAll(async () => {
  await backend?.kill();
});

const status = (page: Page) => page.locator(".race-header .pill[class*='status-']");
const clock = (page: Page) => page.locator(".race-header .clock");
const towerCodes = (page: Page) => page.locator(".tower tbody tr .code").allTextContents();
const row = (page: Page, code: string) => page.locator(".tower tbody tr").filter({ hasText: code });
const button = (page: Page, name: string) => page.getByRole("button", { name, exact: true });

test("play, pause, save, kill the backend, reload, continue, resume, fork", async ({ page }) => {
  test.skip(!hasMonzaData(), `needs fetched Monza data (${MONZA_DATA}): scripts/fetch_event.py`);
  await backend.start();

  await test.step("create a world in the library", async () => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "World Library" })).toBeVisible();
    await expect(page.getByText("No saved worlds yet.")).toBeVisible();
    await page.locator(".new-race select").first().selectOption("2024_italian_gp");
    await page.getByPlaceholder("random").fill("7");
    await page.locator(".new-race select").nth(1).selectOption("25");
    await button(page, "Create").click();
    await expect(status(page)).toHaveText("CREATED");
    await expect(page.locator(".tower tbody tr")).toHaveCount(20);
    await expect(page.locator(".pill.mode")).toHaveText("Alternate simulation");
  });

  await test.step("play: the race clock moves; pause: it stops", async () => {
    await button(page, "Start").click();
    await expect(status(page)).toHaveText("RUNNING");
    await expect(clock(page)).not.toHaveText("0:00");
    await button(page, "Pause").click();
    await expect(status(page)).toHaveText("PAUSED");
    const frozen = await clock(page).textContent();
    await page.waitForTimeout(1500);
    await expect(clock(page)).toHaveText(frozen ?? "");
  });

  let savedClock = "";
  let savedOrder: string[] = [];
  await test.step("step 2 minutes and save as a named checkpoint", async () => {
    await button(page, "+1 min").click();
    await expect(button(page, "+1 min")).toBeEnabled();
    await button(page, "+1 min").click();
    await expect(button(page, "+1 min")).toBeEnabled();
    await button(page, "Save as…").click();
    const dialog = page.getByRole("dialog");
    await expect(dialog.locator("input")).toBeFocused(); // slice-3 bug: focus was on ✕
    await dialog.locator("input").fill(SAVE_NAME);
    await dialog.locator("input").press("Enter");
    await expect(page.locator(".controls .notice")).toContainText(`“${SAVE_NAME}”`);
    savedClock = (await clock(page).textContent()) ?? "";
    savedOrder = await towerCodes(page);
    expect(savedOrder).toHaveLength(20);
  });

  await test.step("kill the backend, start a new one, reload the browser", async () => {
    await backend.kill();
    await backend.start();
    expect((await backend.health()).live_world_id).toBeNull();
    await page.reload();
    await expect(page.getByRole("heading", { name: "World Library" })).toBeVisible();
    await expect(page.locator(".world-card")).toHaveCount(1);
    await expect(page.locator(".world-card")).toContainText(SAVE_NAME);
  });

  await test.step("continue: the same race time, order and PAUSED state come back", async () => {
    await button(page, "Continue").click();
    await expect(status(page)).toHaveText("PAUSED");
    await expect(clock(page)).toHaveText(savedClock);
    expect(await towerCodes(page)).toEqual(savedOrder);
    await expect(page.locator(".pill.branch")).toHaveCount(0);
  });

  await test.step("resume and pause again", async () => {
    await button(page, "Play").click();
    await expect(status(page)).toHaveText("RUNNING");
    await expect(clock(page)).not.toHaveText(savedClock);
    await button(page, "Pause").click();
    await expect(status(page)).toHaveText("PAUSED");
  });

  await test.step("fork: HAM boxes for hards on a branch", async () => {
    await row(page, "HAM").click();
    await expect(page.locator(".driver-panel .dp-name")).toContainText("Lewis Hamilton");
    await expect(page.locator(".driver-panel .source-editorial_assumption").first()).toBeVisible();
    await button(page, "Fork…").click();
    const dialog = page.getByRole("dialog");
    await expect(dialog.locator("select").last()).toHaveValue("lewis_hamilton_2024");
    await dialog.getByRole("button", { name: "Fork", exact: true }).click();
    await expect(page.locator(".pill.branch")).toBeVisible();
    await expect(row(page, "HAM")).toContainText("BOX");
    await expect(page.locator(".tab-body .notice")).toContainText("The original timeline is unchanged");
  });

  await test.step("the main timeline is untouched: load it back", async () => {
    await page.getByRole("tab", { name: "Timelines" }).click();
    const timelines = page.locator(".timelines > li");
    await expect(timelines).toHaveCount(2);
    const main = timelines.first();
    await expect(main).toContainText("Main timeline");
    await expect(main).toContainText(SAVE_NAME);
    await main.locator(".snaps li").first().getByRole("button", { name: "Load" }).click();
    await expect(page.locator(".pill.branch")).toHaveCount(0);
    await expect(row(page, "HAM")).not.toContainText("BOX");
    await expect(page.locator(".tab-body .notice")).toHaveText(/^Loaded “/); // no new branch
  });
});
