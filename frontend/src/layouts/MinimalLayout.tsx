import { Outlet } from 'react-router-dom'

export function MinimalLayout() {
  return (
    <div className="min-h-screen">
      <Outlet />
    </div>
  )
}
