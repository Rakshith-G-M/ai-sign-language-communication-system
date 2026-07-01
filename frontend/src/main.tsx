import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import { AppProviders } from '@/app/providers/AppProviders'
import { ThemeInitializer } from '@/app/providers/ThemeInitializer'
import { router } from '@/app/router/routes'
import '@/styles/globals.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppProviders>
      <ThemeInitializer />
      <RouterProvider router={router} />
    </AppProviders>
  </StrictMode>,
)
