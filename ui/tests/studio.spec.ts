import { test, expect } from '@playwright/test'
import { consumeEvents } from '../src/api'
import { readFile } from 'node:fs/promises'

test('workspace selection is tenant scoped and suggestions follow its domain', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.getByText('Service online', { exact: true })).toBeVisible()
  await expect(page.locator('#workspace')).toHaveValue('1')
  await expect(page.getByText('8 tables in scope')).toBeVisible()
  await page.locator('#tenant').selectOption('2')
  await expect(page.locator('#workspace')).toHaveValue('2')
  await expect(
    page.getByRole('button', { name: /Holdings overview/ }),
  ).toBeVisible()
  await page.getByRole('button', { name: /Holdings overview/ }).click()
  await expect(page.getByLabel('Your analysis question')).toHaveValue(
    'What is the total current holding value by AMC?',
  )
})

test('real SSE shows live stages before the final answer and retains every outcome', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.goto('/')
  await expect(page.locator('#workspace')).toHaveValue('1')
  await page
    .getByLabel('Your analysis question')
    .fill('How many registered customers are there?')
  await page.getByRole('button', { name: 'Run analysis' }).click()
  await expect(
    page
      .getByLabel('Analysis journey')
      .getByText('Live analysis', { exact: true }),
  ).toBeVisible()
  await expect(page.locator('.journey-time')).not.toContainText('-')
  await expect(page.getByLabel('Final answer')).toHaveCount(0)
  await expect(page.locator('#tenant')).toBeDisabled()
  await expect(page.getByLabel('Final answer')).toBeVisible()
  await expect(page.getByLabel('Final answer')).toContainText(
    '20 registered customers',
  )
  await expect(page.locator('.stage-row.completed')).toHaveCount(8)
  await page
    .getByRole('button', { name: 'Run analysis: completed', exact: true })
    .click()
  await page.getByRole('button', { name: /^Activity/ }).click()
  await expect(
    page.getByText('SQL_STAGE 1/2: completed; rows=20', { exact: true }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Outcome', exact: true }).click()
  await expect(page.getByText('Result rows', { exact: true })).toBeVisible()
  await page
    .getByRole('button', { name: /Connect workspace: completed/ })
    .click()
  await expect(page.locator('.key-values')).toContainText('Tenant 1')
  await page
    .getByLabel('Final answer')
    .getByRole('button', { name: 'Export' })
    .click()
  await page
    .locator('nav')
    .getByRole('button', { name: /Analysis history/ })
    .click()
  await expect(
    page
      .locator('.history-list')
      .getByRole('button', { name: /How many registered customers are there/ }),
  ).toBeVisible()
  expect(errors).toEqual([])
})

test('failure keeps completed evidence and does not present a fabricated final answer', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.locator('#workspace')).toHaveValue('1')
  await page.getByLabel('Your analysis question').fill('Fail this analysis')
  await page.getByRole('button', { name: 'Run analysis' }).click()
  await expect(page.getByRole('alert')).toContainText(
    'The selected period could not be validated',
  )
  await expect(page.getByLabel('Final answer')).toHaveCount(0)
  await expect(
    page.getByRole('button', { name: 'Run analysis: failed', exact: true }),
  ).toBeVisible()
  await expect(page.locator('.stage-row.completed')).toHaveCount(5)
})

test('stop cancels the stream and re-enables the question controls', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.locator('#workspace')).toHaveValue('1')
  await page.getByLabel('Your analysis question').fill('Slow analysis')
  await page.getByRole('button', { name: 'Run analysis' }).click()
  await expect(
    page.getByRole('button', { name: 'Stop', exact: true }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Stop', exact: true }).click()
  await expect(
    page.getByText(
      'Analysis stopped. Completed stage outcomes are still available.',
    ),
  ).toBeVisible()
  await expect(page.locator('#tenant')).toBeEnabled()
  await expect(page.getByLabel('Final answer')).toHaveCount(0)
})

test('mobile layout has no horizontal overflow and help supports keyboard dismissal', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await expect(page.locator('#workspace')).toHaveValue('1')
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true)
  await page.getByLabel('Open navigation').click()
  await page
    .locator('nav')
    .getByRole('button', { name: /Data catalog/ })
    .click()
  await expect(
    page.getByRole('heading', { name: 'Your data, in context.' }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Help & guide' }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).toHaveCount(0)
})

test('SSE parser supports split UTF-8, CRLF, multi-line data and rejects an interrupted stream', async () => {
  const input =
    ': heartbeat\r\n\r\ndata: {"kind":"result",\r\ndata: "answer":"₹12"}\r\n\r\n'
  const bytes = new TextEncoder().encode(input)
  const stream = new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += 2)
        controller.enqueue(bytes.slice(i, i + 2))
      controller.close()
    },
  })
  const seen: unknown[] = []
  await consumeEvents(new Response(stream), (event) => seen.push(event))
  expect(seen).toEqual([{ kind: 'result', answer: '₹12' }])
  await expect(
    consumeEvents(new Response('data: {"kind":"ready"}\n\n'), () => {}),
  ).rejects.toThrow('before the analysis finished')
})

test('all SQL statements and exact sandbox source are browsable and downloadable', async ({
  page,
  context,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await page.goto('/')
  await expect(page.locator('#workspace')).toHaveValue('1')
  const responsePromise = page.waitForResponse((response) =>
    response.url().endsWith('/api/questions/stream'),
  )
  await page
    .getByLabel('Your analysis question')
    .fill('Show Python execution evidence')
  await page.getByRole('button', { name: 'Run analysis', exact: true }).click()
  await expect(page.getByLabel('Final answer')).toBeVisible()
  const response = await responsePromise
  const events = (await response.text())
    .split('\n')
    .filter((line) => line.startsWith('data: '))
    .map((line) => JSON.parse(line.slice(6)))
  const evidence = events.find(
    (event) => event.stage === 'execute' && event.status === 'completed',
  ).outcome
  await page
    .getByRole('button', { name: 'Run analysis: completed', exact: true })
    .click()
  await expect(
    page.getByRole('tab', { name: 'SQL 3', exact: true }),
  ).toHaveAttribute('aria-selected', 'true')
  await expect(page.locator('.execution-code-card.sql')).toHaveCount(3)
  await page.getByRole('button', { name: 'Expand all', exact: true }).click()
  for (const title of ['Statement 01', 'Statement 02', 'Statement 03']) {
    await expect(
      page.getByLabel(`${title} source code`, { exact: true }),
    ).toBeVisible()
  }
  await expect(
    page.locator('.execution-code-card.sql .token.keyword').first(),
  ).toBeVisible()
  await page
    .getByRole('button', { name: 'Copy Statement 03', exact: true })
    .click()
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(
    evidence.queries[2],
  )
  await page.getByRole('tab', { name: 'Python 1', exact: true }).click()
  await expect(
    page.getByLabel('Python · py3 source code', { exact: true }),
  ).toBeVisible()
  await expect(
    page.getByLabel('Python · py3 source code', { exact: true }),
  ).toContainText('from decimal import Decimal')
  await expect(
    page.getByText('Monty sandbox · Inputs: query_1', { exact: true }),
  ).toBeVisible()
  await page
    .getByRole('button', { name: 'Copy Python · py3', exact: true })
    .click()
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(
    evidence.calculations[0].code,
  )
  const downloadPromise = page.waitForEvent('download')
  await page
    .getByRole('button', { name: 'Download Python · py3', exact: true })
    .click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toBe('analysis-py3.py')
  expect(await readFile((await download.path())!, 'utf8')).toBe(
    evidence.calculations[0].code,
  )
  await page.getByRole('tab', { name: 'Results 1', exact: true }).click()
  await expect(page.locator('.evidence-dataset')).toContainText('amount_inr')
  await page.getByText('Calculation output · py3', { exact: true }).click()
  await expect(page.locator('.calculation-result pre')).toContainText('13.25')
})

test('SQL-only analyses do not show Python and reused results are labelled accurately', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.locator('#workspace')).toHaveValue('1')
  await page.getByLabel('Your analysis question').fill('Count customers')
  await page.getByRole('button', { name: 'Run analysis', exact: true }).click()
  await expect(page.getByLabel('Final answer')).toBeVisible()
  await page
    .getByRole('button', { name: 'Run analysis: completed', exact: true })
    .click()
  await expect(page.getByRole('tab', { name: /^Python/ })).toHaveCount(0)
  await page
    .getByLabel('Your analysis question')
    .fill('Show Python reuse evidence')
  await page.getByRole('button', { name: 'Run analysis', exact: true }).click()
  await expect(page.getByLabel('Final answer')).toBeVisible()
  await page
    .getByRole('button', { name: 'Run analysis: completed', exact: true })
    .click()
  await page.getByRole('tab', { name: 'Python 2', exact: true }).click()
  await expect(
    page
      .getByRole('article', { name: 'Python · py3', exact: true })
      .getByText('Executed', { exact: true }),
  ).toBeVisible()
  await expect(
    page
      .getByRole('article', { name: 'Python · py4', exact: true })
      .getByText('Reused', { exact: true }),
  ).toBeVisible()
  await expect(
    page.getByLabel('Python · py4 source code', { exact: true }),
  ).toContainText('from decimal import Decimal')
})

test('execution code remains contained on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await expect(page.locator('#workspace')).toHaveValue('1')
  await page
    .getByLabel('Your analysis question')
    .fill('Show Python execution evidence')
  await page.getByRole('button', { name: 'Run analysis', exact: true }).click()
  await expect(page.getByLabel('Final answer')).toBeVisible()
  await page
    .getByRole('button', { name: 'Run analysis: completed', exact: true })
    .click()
  await page.getByRole('tab', { name: 'Python 1', exact: true }).click()
  await expect(
    page.getByLabel('Python · py3 source code', { exact: true }),
  ).toBeVisible()
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true)
})
