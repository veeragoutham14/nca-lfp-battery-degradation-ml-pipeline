import { mkdir } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const frontendDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const projectRoot = path.resolve(frontendDir, '..')
const outputDir = path.join(projectRoot, 'var', 'screenshots')
const browserRoot = path.join(projectRoot, '.tools', 'playwright')
const dashboardUrl = process.env.BATTRYML_DASHBOARD_URL ?? 'http://127.0.0.1:5173'
const username = process.env.BATTRYML_QA_USERNAME ?? 'admin'
const password = process.env.BATTRYML_QA_PASSWORD ?? 'battryml'

process.env.PLAYWRIGHT_BROWSERS_PATH ??= browserRoot
const { chromium } = await import('playwright-core')
await mkdir(outputDir, { recursive: true })

const browser = await chromium.launch({ headless: true })
const errors = []
const layouts = {}

function observe(page, name) {
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(`${name}: ${message.text()}`)
  })
  page.on('pageerror', (error) => errors.push(`${name}: ${error.message}`))
}

async function capture(page, name) {
  await page.waitForTimeout(500)
  layouts[name] = await page.evaluate(() => ({
    title: document.title,
    bodyWidth: document.body.scrollWidth,
    viewportWidth: window.innerWidth,
    bodyHeight: document.body.scrollHeight,
    canvasCount: document.querySelectorAll('canvas').length,
    heading: document.querySelector('h1')?.textContent ?? '',
  }))
  await page.screenshot({ path: path.join(outputDir, `${name}.png`), fullPage: true })
}

async function captureLogin(name, width, height) {
  const context = await browser.newContext({ viewport: { width, height } })
  const page = await context.newPage()
  observe(page, name)
  await page.goto(dashboardUrl, { waitUntil: 'networkidle' })
  await page.getByRole('heading', { name: 'Operator sign in' }).waitFor()
  await capture(page, name)
  await context.close()
}

await captureLogin('login-desktop', 1_440, 1_000)
await captureLogin('login-mobile', 390, 844)

const context = await browser.newContext({ viewport: { width: 1_440, height: 1_000 } })
const page = await context.newPage()
let seasonalRunId = ''
observe(page, 'workspace')
await page.goto(dashboardUrl, { waitUntil: 'networkidle' })
await page.getByLabel('Username').fill(username)
await page.getByLabel('Password').fill(password)
await page.getByRole('button', { name: 'Sign in' }).click()
await page.getByRole('heading', { name: 'Select an analysis workspace' }).waitFor()
await capture(page, 'chemistry-select-desktop')
await page.getByRole('button', { name: /^NCA/ }).click()
await page.getByRole('heading', { name: 'NCA battery data operations' }).waitFor()
await capture(page, 'home-desktop')

await page.getByRole('button', { name: 'Extraction', exact: true }).click()
await page.getByRole('heading', { name: 'Raw telemetry' }).waitFor()
await capture(page, 'extraction-desktop')

await page.getByRole('button', { name: 'Machine learning', exact: true }).click()
await page.getByText('Processing workflow').first().waitFor().catch(() => undefined)
await capture(page, 'ml-desktop')

await page.getByRole('button', { name: 'BMS SOH events', exact: true }).click()
await page.getByRole('heading', { name: 'BMS SOH recalibration events' }).waitFor()
await page.locator('.recalibration-chart-panel canvas').waitFor().catch(() => undefined)
await capture(page, 'bms-soh-events-desktop')

await page.getByRole('button', { name: 'Seasonal profiles', exact: true }).click()
await page.getByRole('heading', { name: 'Seasonal degradation and recalibration profiles' }).waitFor()
await page.locator('.seasonal-profile-panel canvas').waitFor().catch(() => undefined)
await capture(page, 'seasonal-profiles-desktop')
for (const tabName of ['Current dynamics', 'Combined exposures']) {
  const tab = page.getByRole('tab', { name: tabName })
  if (await tab.count()) {
    await tab.click()
    await capture(page, `seasonal-${tabName.toLowerCase().replaceAll(' ', '-')}-desktop`)
  }
}

const firstReport = page.locator('.run-row').first()
if (await firstReport.count()) {
  await firstReport.click()
  await page.locator('.results-page').waitFor()
  await capture(page, 'report-desktop')
} else {
  const runs = await page.evaluate(async () => {
    const response = await fetch('/api/runs?serial=100000017')
    return response.ok ? response.json() : []
  })
  if (runs.length) {
    await page.goto(`${dashboardUrl}/#/run/${encodeURIComponent(runs[0].run_id)}`, {
      waitUntil: 'networkidle',
    })
    await page.locator('.results-page').waitFor()
    await capture(page, 'report-desktop')
  }
}

const seasonalRun = await page.evaluate(async () => {
  const response = await fetch('/api/runs?model_family=seasonal_charging_stress_profile&limit=100')
  if (!response.ok) return null
  const runs = await response.json()
  for (const run of runs) {
    const seriesResponse = await fetch(`/api/runs/${encodeURIComponent(run.run_id)}/series`)
    if (!seriesResponse.ok) continue
    const series = await seriesResponse.json()
    const supportsFullSoc = (series.seasonal_profile ?? []).some(
      (point) => point.soc_40_60_hours != null || point.soc_unknown_hours != null,
    )
    if (supportsFullSoc) return run
  }
  return runs[0] ?? null
})
if (seasonalRun) {
  seasonalRunId = seasonalRun.run_id
  await page.goto(`${dashboardUrl}/#/run/${encodeURIComponent(seasonalRunId)}`, {
    waitUntil: 'networkidle',
  })
  await page.locator('.seasonal-profile-panel canvas').waitFor()
  await capture(page, 'seasonal-report-desktop')
  const fullSocTab = page.getByRole('tab', { name: 'Full SOC' })
  if (await fullSocTab.count()) {
    await fullSocTab.click()
    await capture(page, 'seasonal-full-soc-desktop')
  }
}

await page.setViewportSize({ width: 390, height: 844 })
await page.goto(`${dashboardUrl}/#/home`, { waitUntil: 'networkidle' })
await capture(page, 'home-mobile')
await page.goto(`${dashboardUrl}/#/extraction`, { waitUntil: 'networkidle' })
await capture(page, 'extraction-mobile')
await page.goto(`${dashboardUrl}/#/ml`, { waitUntil: 'networkidle' })
await capture(page, 'ml-mobile')
await page.getByRole('button', { name: 'BMS SOH events', exact: true }).click()
await page.getByRole('heading', { name: 'BMS SOH recalibration events' }).waitFor()
await page.locator('.recalibration-chart-panel canvas').waitFor().catch(() => undefined)
await capture(page, 'bms-soh-events-mobile')
await page.getByRole('button', { name: 'Seasonal profiles', exact: true }).click()
await page.getByRole('heading', { name: 'Seasonal degradation and recalibration profiles' }).waitFor()
await capture(page, 'seasonal-profiles-mobile')
if (seasonalRunId) {
  await page.goto(`${dashboardUrl}/#/run/${encodeURIComponent(seasonalRunId)}`, {
    waitUntil: 'networkidle',
  })
  await page.locator('.seasonal-profile-panel canvas').waitFor()
  await capture(page, 'seasonal-report-mobile')
  const fullSocTab = page.getByRole('tab', { name: 'Full SOC' })
  if (await fullSocTab.count()) {
    await fullSocTab.click()
    await capture(page, 'seasonal-full-soc-mobile')
  }
}

await context.close()
await browser.close()

const report = { dashboardUrl, layouts, errors }
console.log(JSON.stringify(report, null, 2))

const hasHorizontalOverflow = Object.values(layouts).some(
  (layout) => layout.bodyWidth > layout.viewportWidth,
)
const missingExpectedHeading = layouts['login-desktop']?.heading !== 'Operator sign in'
  || layouts['chemistry-select-desktop']?.heading !== 'Select an analysis workspace'
  || layouts['home-desktop']?.heading !== 'NCA battery data operations'
  || layouts['extraction-desktop']?.heading !== 'Raw telemetry'

if (errors.length || hasHorizontalOverflow || missingExpectedHeading) {
  process.exitCode = 1
}
