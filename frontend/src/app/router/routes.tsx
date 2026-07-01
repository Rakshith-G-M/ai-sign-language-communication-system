import { Navigate, createBrowserRouter } from 'react-router-dom'
import { AppShellLayout } from '@/layouts/AppShellLayout'
import { MinimalLayout } from '@/layouts/MinimalLayout'
import { DashboardPage } from '@/pages/DashboardPage'
import { SystemPage } from '@/pages/SystemPage'
import { OnboardingPage } from '@/pages/OnboardingPage'
import { NotFoundPage } from '@/pages/NotFoundPage'
import { ONBOARDING_STORAGE_KEY } from '@/lib/constants'

function OnboardingGuard({ children }: { children: React.ReactNode }) {
  const complete = localStorage.getItem(ONBOARDING_STORAGE_KEY) === 'true'
  if (!complete) {
    return <Navigate to="/onboarding" replace />
  }
  return children
}

function OnboardingRedirect({ children }: { children: React.ReactNode }) {
  const complete = localStorage.getItem(ONBOARDING_STORAGE_KEY) === 'true'
  if (complete) {
    return <Navigate to="/dashboard" replace />
  }
  return children
}

export const router = createBrowserRouter([
  {
    element: <MinimalLayout />,
    children: [
      {
        path: '/onboarding',
        element: (
          <OnboardingRedirect>
            <OnboardingPage />
          </OnboardingRedirect>
        ),
      },
    ],
  },
  {
    element: (
      <OnboardingGuard>
        <AppShellLayout />
      </OnboardingGuard>
    ),
    children: [
      { path: '/', element: <Navigate to="/dashboard" replace /> },
      { path: '/dashboard', element: <DashboardPage /> },
      { path: '/system', element: <SystemPage /> },
    ],
  },
  {
    path: '*',
    element: <NotFoundPage />,
  },
])
