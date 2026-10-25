import { spawnSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const frontendDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const projectRoot = path.resolve(frontendDir, '..')
const browserRoot = path.join(projectRoot, '.tools', 'playwright')
const cli = path.join(frontendDir, 'node_modules', 'playwright-core', 'cli.js')

const result = spawnSync(process.execPath, [cli, 'install', 'chromium'], {
  cwd: frontendDir,
  env: { ...process.env, PLAYWRIGHT_BROWSERS_PATH: browserRoot },
  stdio: 'inherit',
})

process.exitCode = result.status ?? 1
