import { BatteryCharging, LockKeyhole, UserRound } from 'lucide-react'
import { FormEvent, useState } from 'react'
import type { Health } from '../types'

interface LoginPageProps {
  health: Health | null
  onLogin: (username: string, password: string) => Promise<void>
}

export function LoginPage({ health, onLogin }: LoginPageProps) {
  const [username, setUsername] = useState('admin')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function submit(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      await onLogin(username.trim(), password)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="login-page">
      <section className="login-surface" aria-labelledby="login-title">
        <header className="login-brand">
          <span className="login-mark"><BatteryCharging size={30} aria-hidden="true" /></span>
          <div>
            <span>BatteryML</span>
            <h1 id="login-title">Operator sign in</h1>
          </div>
        </header>
        <form className="login-form" onSubmit={submit}>
          <label className="field input-with-icon">
            <span>Username</span>
            <span className="input-control">
              <UserRound size={16} aria-hidden="true" />
              <input
                autoComplete="username"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                required
              />
            </span>
          </label>
          <label className="field input-with-icon">
            <span>Password</span>
            <span className="input-control">
              <LockKeyhole size={16} aria-hidden="true" />
              <input
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
              />
            </span>
          </label>
          {error && <div className="form-error">{error}</div>}
          <button className="button primary login-button" type="submit" disabled={busy}>
            {busy ? 'Signing in...' : 'Sign in'}
          </button>
        </form>
        <footer className="login-status">
          <span className={`status-dot ${health?.status === 'ok' ? 'status-online' : ''}`} />
          {health?.status === 'ok' ? 'Local API ready' : 'Waiting for local API'}
        </footer>
      </section>
    </main>
  )
}
