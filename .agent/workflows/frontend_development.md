# Workflow: Frontend Development

Follow these steps to add new features, components, or pages to the Next.js frontend.

## 1. Create a New Component
- Place shared UI components in `frontend/src/components/`.
- Use functional components with generic CSS or Tailwind if applicable.
- Keep components small and focused on one task.

## 2. Add or Modify Pages
- Pages are located in `frontend/src/app/` (using App Router).
- Use `page.tsx` for the main page logic.

## 3. Connecting to the Backend
- Use the `API_URL` environment variable (mapped to `http://localhost:5001`).
- Fetch data in `useEffect` or use Server Components where appropriate.
- Ensure all API calls handle errors gracefully.

## 4. Verification
- Run the dev server: `npm run dev` in the `frontend` directory.
- Verify the UI layout and data responsiveness.
- Check the browser console for any React warnings or errors.
