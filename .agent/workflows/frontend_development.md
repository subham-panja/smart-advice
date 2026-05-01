# Workflow: Frontend Development

Follow these steps to add new features, components, or pages to the Next.js frontend.

## 1. Project Structure
- **Framework**: Next.js 15.5 with App Router
- **React**: React 19
- **Styling**: Tailwind CSS v4
- **Charts**: Chart.js 4 with `react-chartjs-2`
- **Icons**: Heroicons (`@heroicons/react`)
- **Tables**: TanStack React Table (`@tanstack/react-table`)
- **UI**: Headless UI (`@headlessui/react`)
- **HTTP**: Axios

## 2. Existing Pages
- `/` - Main dashboard with charts, analysis trigger, and recommendations
- `/about` - About page with feature overview and system status
- `/analysis` - Detailed analysis page with configuration
- `/recommendations` - Recommendation listing with DataTable
- `/fo-analysis` - Futures & Options analysis page
- `/fo-recommendations` - F&O-specific recommendations
- `/settings` - Settings and configuration page

## 3. Create a New Component
- Place shared UI components in `frontend/src/app/components/`.
- Existing components: `Navbar`, `Sidebar`, `DataTable`, `ThemeToggle`, `ApiTest`, `Terminal`, `MainContent`.
- Use functional components with Tailwind CSS v4.
- Keep components small and focused on one task.

## 4. Add or Modify Pages
- Pages are located in `frontend/src/app/` (using App Router).
- Use `page.tsx` for the main page logic.
- Follow the existing pattern with `'use client'` for interactive pages.

## 5. Connecting to the Backend
- Use the `NEXT_PUBLIC_API_URL` environment variable (default: `http://127.0.0.1:5001`).
- API utilities are in `frontend/src/lib/api.ts`.
- Existing endpoints:
  - `GET /` - Health check
  - `GET /recommendations` - Fetch all stock recommendations
  - `POST /trigger-analysis` - Start stock analysis (requires `group` in body)
  - `GET /analyze_stock/<symbol>` - Single stock analysis
  - `GET /symbol-groups` - Get available symbol groups
- Fetch data in `useEffect` or use Server Components where appropriate.
- Ensure all API calls handle errors gracefully (see `api.ts` for error handling patterns).

## 6. Verification
- Run the dev server: `npm run dev` in the `frontend` directory.
- Verify the UI layout and data responsiveness.
- Check the browser console for any React warnings or errors.
- Run tests: `npm run test` (Jest), `npm run test:e2e` (Playwright).
